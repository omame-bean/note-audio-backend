from fastapi import FastAPI, HTTPException, Request, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from sse_starlette.sse import EventSourceResponse
from pydantic import BaseModel
import tempfile
import os
from moviepy.editor import *
from PIL import Image, ImageDraw, ImageFont
import requests
from io import BytesIO
from openai import OpenAI
import json
import re
import time
import numpy as np
import cv2
import logging
import ffmpeg
import traceback
from pydub import AudioSegment
import random
import textwrap
from os import getenv
import asyncio
import aiohttp
import uuid
import shutil
import shlex
import subprocess

# ロギングの基本設定
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

from dotenv import load_dotenv
load_dotenv()

app = FastAPI()

@app.get("/check-ffmpeg")
async def check_ffmpeg():
    try:
        result = subprocess.run(["ffmpeg", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode == 0:
            ffmpeg_version = result.stdout.split('\n')[0]
            return {"ffmpeg_version": ffmpeg_version}
        else:
            return {"error": "FFmpegが見つかりませんでした。"}
    except Exception as e:
        return {"error": str(e)}

# CORSミドルウェアを追加
FRONTEND_URL = getenv("FRONTEND_URL", "http://localhost:3000")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_URL],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ログ設定
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# クライアントごとの進捗を管理する辞書
client_progress = {}


# OpenAIクライアントの初期化
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class VideoRequest(BaseModel):
    client_id: str  # クライアントを一意に識別するID
    note_content: str

class VideoResponse(BaseModel):
    video_url: str

async def progress_generator(request: Request):
    progress_steps = [
        "テキスト解析",
        "音声合成",
        "画像生成",
        "動画編集",
        "最終出力"
    ]
    for step in progress_steps:
        if await request.is_disconnected():
            break
        yield json.dumps({"step": step, "status": "in-progress", "message": f"{step}中..."})
        await asyncio.sleep(1)  # シミュレーションのための遅延
        yield json.dumps({"step": step, "status": "completed", "message": f"{step}完了"})
        await asyncio.sleep(0.5)


@app.post("/generate-video", response_model=VideoResponse)
async def generate_video(video_request: VideoRequest, background_tasks: BackgroundTasks):
    client_id = video_request.client_id
    note_content = video_request.note_content
    client_progress[client_id] = []
    background_tasks.add_task(run_video_generation, client_id, note_content)
    logger.info(f"クライアント {client_id} の動画生成を開始しました。ノート内容: {note_content[:100]}...")
    return VideoResponse(video_url="")

@app.get("/events/{client_id}")
async def events(client_id: str):
    async def event_generator():
        logger.info(f"クライアント {client_id} がイベントのために接続しました。")
        last_sent_index = 0
        retry_count = 0
        max_retries = 5
        while retry_count < max_retries:
            try:
                if client_id not in client_progress:
                    await asyncio.sleep(1)
                    retry_count += 1
                    continue
                
                steps = client_progress[client_id]
                new_steps = steps[last_sent_index:]
                for step in new_steps:
                    yield {"data": step}
                    last_sent_index += 1
                
                if any(json.loads(step).get("step") == "最終出力" for step in new_steps):
                    logger.info(f"クライアント {client_id} の処理が完了しました。")
                    break
                
                await asyncio.sleep(1)
            except Exception as e:
                logger.error(f"イベント生成中にエラーが発生しました: {str(e)}")
                await asyncio.sleep(1)
                retry_count += 1
        
        if retry_count >= max_retries:
            logger.error(f"クライアント {client_id} の最大再試行回数に達しました。")
    
    return EventSourceResponse(event_generator())

async def run_video_generation(client_id: str, note_content: str):
    temp_dir = tempfile.mkdtemp(prefix="video_gen_")
    try:
        async def update_progress(step, status, message=None, video_url=None):
            progress = {
                "step": step,
                "status": status,
                "message": message,
                "video_url": video_url
            }
            client_progress[client_id].append(json.dumps(progress))
            logger.info(f"クライアント {client_id}: {progress}")
            await asyncio.sleep(0.1)  # 短い遅延を追加

        # テキスト解析
        await update_progress("テキスト解析", "in-progress", "テキスト解析中...")
        script = await asyncio.to_thread(generate_script, note_content)
        await update_progress("テキスト解析", "completed", "台詞が正常に生成されました")

        # 背景選択
        await update_progress("背景選択", "in-progress", "背景動画を選択中...")
        background_video = await asyncio.to_thread(select_background, note_content)
        await update_progress("背景選択", "completed", f"選択された背景動画: {background_video}")

        # 背景音楽選択
        await update_progress("背景音楽選択", "in-progress", "背景音楽を選択中...")
        background_music = await asyncio.to_thread(select_background_music, note_content)
        await update_progress("背景音楽選択", "completed", f"選択された背景音楽: {background_music}")

        # 画像生成と音声合成を並列で実行
        await update_progress("画像生成と音声合成", "in-progress", "画像生成と音声合成を開始...")
        images_task = asyncio.create_task(generate_images(script))
        audio_task = asyncio.create_task(generate_audio(script))

        # 両方のタスクが完了するまで待機
        images, (audio_clips, durations) = await asyncio.gather(images_task, audio_task)

        await update_progress("画像生成と音声合成", "completed", "画像生成と音声合成が完了しました")

        # 動画編集
        await update_progress("動画編集", "in-progress", "動画編集中...")
        image_display_times = calculate_image_display_times(durations)
        video_filename = await asyncio.to_thread(create_video, script, images, audio_clips, background_video, background_music, image_display_times, temp_dir)
        await update_progress("動画編集", "completed", f"動画が正確に作成されました: {video_filename}")

        # 最終出力
        video_url = f"/download-video/{os.path.basename(video_filename)}"
        await update_progress("最終出力", "completed", "動画生成が完了しました", video_url)

        # 接続終了メッセージを送信せず、クライアントからの切断を待つ
        while True:
            await asyncio.sleep(10)  # 10秒ごとにチェック
            if client_id not in client_progress:
                break

    except Exception as e:
        error_message = str(e)
        if isinstance(e, ffmpeg._run.Error):
            error_log_path = os.path.join(temp_dir, 'ffmpeg_error.log')
            if os.path.exists(error_log_path):
                with open(error_log_path, 'r') as f:
                    error_content = f.read()
                error_message += f"\nFFmpeg detailed error log:\n{error_content}"
        logger.error(f"Error occurred for client {client_id}: {error_message}")
        await send_error_to_client(client_id, error_message)
        raise
    finally:
        # 一時ディレクトリの削除
        if os.path.exists(temp_dir):
            import shutil
            shutil.rmtree(temp_dir)
        # client_progress からの削除はクライアント側で行うため、ここでは削除しない
        pass

# 画像表示時間を計算する関数（必要に応じて調整）
def calculate_image_display_times(durations):
    image_display_times = []
    cumulative_time = 0.0
    for i in range(0, len(durations), 3):
        group_durations = durations[i:i+3]
        start_time = cumulative_time
        display_duration = sum(group_durations)
        end_time = start_time + display_duration
        image_display_times.append((start_time, end_time))
        cumulative_time = end_time
    return image_display_times

@app.get("/download-video/{video_filename}")
async def download_video(video_filename: str, client_id: str = Query(...)):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    video_path = os.path.join(current_dir, video_filename)
    logger.debug(f"Requested video path: {video_path}")
    logger.debug(f"File exists: {os.path.exists(video_path)}")
    if os.path.exists(video_path):
        # クライアントの進捗情報を削除
        if client_id in client_progress:
            del client_progress[client_id]
        
        return FileResponse(
            video_path,
            media_type="video/mp4",
            filename=video_filename
        )
    else:
        raise HTTPException(status_code=404, detail=f"動画が見つかりません: {video_path}")

def generate_script(note_content: str) -> list:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "ノート内容から30秒前後の動画用の台詞を成してください。各シーンを辞書形式で返してください。各シーンは'scene_number'、'description'、'script'のキーを持つ必要があります。返答は必ず有効なJSONフォーマットにしてください。全て日本語でいてください。"},
            {"role": "user", "content": note_content}
        ]
    )
    try:
        content = response.choices[0].message.content
        content = re.search(r'\{.*\}', content, re.DOTALL)
        if content:
            content = content.group()
        else:
            raise ValueError("有効なJSONが見つかりませんでした。")
        
        parsed_content = json.loads(content)
        
        if "scenes" in parsed_content:
            scenes = parsed_content["scenes"]
        else:
            scenes = parsed_content
        
        for scene in scenes:
            if "scene_number" not in scene:
                scene["scene_number"] = scenes.index(scene) + 1
            if "description" not in scene:
                scene["description"] = f"シーン {scene['scene_number']}"
            if "script" not in scene and "dialogue" in scene:
                scene["script"] = scene.pop("dialogue")
        
        return scenes
    except json.JSONDecodeError as e:
        print(f"JSONデコードエラ: {e}")
        print(f"APIレスポンス: {content}")
        raise ValueError("APIからの応答を解析できませんでした。")
    except Exception as e:
        print(f"予期せぬエラー: {e}")
        print(f"APIレスポンス: {content}")
        raise ValueError(f"APIからの応答の処理中にエラーが発生しました: {str(e)}")

async def generate_images(script: list) -> list:
    images = []
    for i in range(0, len(script), 3):
        group_scenes = script[i:i+3]
        combined_script = "\n".join([f"シーン{scene['scene_number']}: {scene['script']}" for scene in group_scenes])
        
        prompt = f"""
        以下の内容を表現するリアルに近いイラストを生成してください。人物はいりません。：
        
        {combined_script}
        
        要件：
        - 必ずリアルに近いイラストスタイルで描いてください。
        - 人物は描かないでください。風景や景色を描いてください。
        - スクリプトの内容に直接関連する要素を含めてください。
        - 背景も含め、シーンの雰囲気を表現してください。
        - 画像の中に文字は絶対に入れないでください。
        
        このイラストは動画のシーンとして使用されます。適切な構図と内容を心がけてください。
        """
        
        logger.debug(f"画像生成プロンプト（グループ {i//3 + 1}）:\n{prompt}")
        
        response = client.images.generate(  # awaitを削除
            model="dall-e-3",
            prompt=prompt,
            n=1,
            size="1024x1024"
        )
        image_url = response.data[0].url
        async with aiohttp.ClientSession() as session:
            async with session.get(image_url) as image_response:
                image_data = await image_response.read()
                img = Image.open(BytesIO(image_data))
                images.append(img)
        
        logger.debug(f"生成された画像のURL（グループ {i//3 + 1}）: {image_url}")

    return images

async def generate_audio(script: list) -> tuple:
    audio_clips = []
    durations = []
    for scene in script:
        response = client.audio.speech.create(  # awaitを削除
            model="tts-1",
            voice="nova",
            input=scene['script']
        )
        
        audio_content = response.content

        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as temp_audio_file:
            temp_audio_file.write(audio_content)
            audio_clips.append(temp_audio_file.name)
        
        duration = await asyncio.to_thread(get_audio_duration, temp_audio_file.name)
        durations.append(duration)
    
    return audio_clips, durations

def select_background(note_content: str) -> str:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    movie_dir = os.path.join(current_dir, "movie")
    
    # movie ディレクトリが存在しない場合は作成
    if not os.path.exists(movie_dir):
        os.makedirs(movie_dir)
        logger.warning(f"'movie'ディレクトリが存在しなかったため、作成しました: {movie_dir}")
    
    backgrounds = [f for f in os.listdir(movie_dir) if f.endswith('.mp4')]
    
    if not backgrounds:
        logger.error("背景動画が見つかりません。'movie'ィレクトリにmp4ファイルを追加してください。")
        raise FileNotFoundError("背景動画が見つかりません。")
    
    prompt = f"""
    以下のノート内容に最も適した背景動画を選んでください。
    選択肢は次の通りです: {', '.join(backgrounds)}
    
    ノート内容:
    {note_content}
    
    最も適切な背景動画のファイル名のみを返してください。
    """
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "あなたは与えられたノート内容に基づいて最適な背景動画を選択する専門家です。"},
            {"role": "user", "content": prompt}
        ]
    )
    
    selected_background = response.choices[0].message.content.strip()
    logger.info(f"選択された背景動画: {selected_background}")
    
    if selected_background not in backgrounds:
        logger.warning(f"選択された背景動画 '{selected_background}' が見つかりません。最初の動画を使用します。")
        selected_background = backgrounds[0]
    
    full_path = os.path.join(movie_dir, selected_background)
    logger.info(f"使用する背景動画の完全パス: {full_path}")
    logger.info(f"背景動画ファイルが存在するか: {os.path.exists(full_path)}")
    
    return full_path

def select_background_music(note_content: str) -> str:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    music_dir = os.path.join(current_dir, "music")
    
    if not os.path.exists(music_dir):
        os.makedirs(music_dir)
        logger.warning(f"'music'ディレクトリが存在しなかったため、作成しました: {music_dir}")
    
    music_files = [f for f in os.listdir(music_dir) if f.endswith(('.mp3', '.wav'))]
    
    if not music_files:
        logger.error("背景音楽が見つかりません。'music'ディレクトリに音楽ファイルを追加してください。")
        raise FileNotFoundError("背景音楽が見つかりません。")
    
    prompt = f"""
    以下のノート内容に最も適した背景音楽を選んでください。
    選択肢は次の通りです: {', '.join(music_files)}
    
    ノート内容:
    {note_content}
    
    最も適切な背景音楽のファイル名のみを返してください。
    """
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "あなたは与えられたノート内容に基づいて最適な背景音楽を選択する専門家です。"},
            {"role": "user", "content": prompt}
        ]
    )
    
    selected_music = response.choices[0].message.content.strip()
    logger.info(f"選択された背景音楽: {selected_music}")
    
    if selected_music not in music_files:
        logger.warning(f"選択された背景音楽 '{selected_music}' が見つかりません。最初の音楽ファイルを使用します。")
        selected_music = music_files[0]
    
    full_path = os.path.join(music_dir, selected_music)
    logger.info(f"使用する背景音楽の完全パス: {full_path}")
    logger.info(f"背景音楽ファイルが存在するか: {os.path.exists(full_path)}")
    
    return full_path

def combine_audio(audio_paths: list, output_path: str) -> str:
    try:
        combined = AudioSegment.empty()
        for audio_path in audio_paths:
            audio = AudioSegment.from_file(audio_path)
            combined += audio
        combined.export(output_path, format='mp3')
        logging.info(f"Combined audio saved to {output_path}")
        return output_path
    except Exception as e:
        logging.error(f"Failed to combine audio: {e}")
        raise

def get_audio_duration(audio_path: str) -> float:
    audio = AudioSegment.from_file(audio_path)
    return len(audio) / 1000.0

def wrap_text(text, max_width=40):
    """
    テキストを指定した最大文字数で折り返します。
    max_widthは仮の値で、フォントサイズや解像度に応じて調整が必要です。
    """
    return '\n'.join(textwrap.wrap(text, width=max_width))

def escape_ffmpeg_text(text: str) -> str:
    """
    FFmpegのdrawtextフィルターで使用するテキストをエスケープする関数
    """
    # バックスラッシュとシングルクォートをエスケープ
    text = text.replace('\\', '\\\\').replace("'", "\\'")
    # カンマとコロンをエスケープ
    text = text.replace(',', '\\,').replace(':', '\\:')
    return text

def create_video(script: list, images: list, audio_clips: list, background_video: str, background_music: str, image_display_times: list, temp_dir: str, orientation='landscape') -> str:
    # ファイルの存在とパーミッションを確認する関数
    def check_file(file_path):
        if os.path.exists(file_path):
            if os.access(file_path, os.R_OK):
                logger.info(f"File exists and is readable: {file_path}")
            else:
                logger.error(f"File exists but is not readable: {file_path}")
        else:
            logger.error(f"File does not exist: {file_path}")

    # 各ファイルを確認
    check_file("/opt/render/project/src/movie/digital_grid.mp4")
    check_file("/tmp/video_gen_ueexpoxa/temp_image_0.png")
    check_file("/tmp/video_gen_ueexpoxa/temp_image_1.png")
    check_file("/opt/render/project/src/mixed_audio_1729187440.mp3")
    check_file("/opt/render/project/src/font.ttf")

    # FFmpegの存在を確認
    try:
        subprocess.run(["ffmpeg", "-version"], check=True, capture_output=True, text=True)
        logger.info("FFmpeg is installed and accessible")
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg is not installed or not accessible: {e}")
    except FileNotFoundError:
        logger.error("FFmpeg command not found")

    # 背景動画の情報を取得
    probe = ffmpeg.probe(background_video)
    video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
    if video_stream is None:
        raise ValueError("背景動画にビデオストリームが見つかりません")
    
    logger.info(f"背景動画情報: {video_stream}")
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    timestamp = int(time.time())
    output_path = os.path.join(current_dir, f"generated_video_{timestamp}.mp4")
    combined_audio_path = os.path.join(current_dir, f"combined_audio_{timestamp}.mp3")

    logger.debug(f"Current directory: {current_dir}")
    logger.debug(f"Output path: {output_path}")
    logger.debug(f"Combined audio path: {combined_audio_path}")
    logger.debug(f"背景動画パス: {background_video} (存在: {os.path.exists(background_video)})")
    logger.debug(f"背景音楽パス: {background_music} (存在: {os.path.exists(background_music)})")

    try:
        combined_audio_path = combine_audio(audio_clips, combined_audio_path)
        total_duration = get_audio_duration(combined_audio_path)

        if total_duration <= 0:
            logger.error("オーディオの長さが0秒以下です。")
            raise ValueError("オーディオの長さが0秒以下です。")

        # 背景音楽の準備
        background_music_audio = AudioSegment.from_file(background_music)
        background_music_audio = background_music_audio - 25  # 音量を30%下げる（-10dB）
        if len(background_music_audio) < total_duration * 1000:
            background_music_audio = background_music_audio * (int(total_duration * 1000 / len(background_music_audio)) + 1)
        background_music_audio = background_music_audio[:int(total_duration * 1000)]
        
        # 音声の音量を90%に下げる
        voice_audio = AudioSegment.from_file(combined_audio_path)
        voice_audio = voice_audio + 0.11  # 音量

        # 音声と背景音楽をミックス
        mixed_audio = voice_audio.overlay(background_music_audio)
        mixed_audio_path = os.path.join(current_dir, f"mixed_audio_{timestamp}.mp3")
        mixed_audio.export(mixed_audio_path, format='mp3')

        # 各シーンごとのテロップ用タイムスタンプを作成
        scene_timestamps = []
        cumulative_time = 0.0
        durations = []
        for audio_path in audio_clips:
            duration = get_audio_duration(audio_path)
            durations.append(duration)
        
        for idx, scene in enumerate(script):
            scene_duration = durations[idx] if idx < len(durations) else 5.0  # デフォルトを5秒とする
            scene_start = cumulative_time
            scene_end = cumulative_time + scene_duration
            wrapped_text = wrap_text(scene['script'], max_width=40)  # テキストを折り返す
            scene_timestamps.append((wrapped_text, scene_start, scene_end))
            cumulative_time = scene_end

        # フォントファイルの絶対パスを取得
        font_path = os.path.join(current_dir, 'font.ttf')
        font_path = os.path.abspath(font_path)  # 絶対パスに変換

        # フォントファイルの存在確認
        if not os.path.exists(font_path):
            logger.error(f"フォントファイルが見つかりません: {font_path}")
            raise FileNotFoundError(f"フォントファイルが見つかりません: {font_path}")
        
        logger.debug(f"Using font file: {font_path}")
        logger.debug(f"フォントパス: {font_path} (存在: {os.path.exists(font_path)})")

        # 1. 背景動画のスケーリングのみを行う
        scaled_bg = ffmpeg.input(background_video, stream_loop=-1, t=total_duration)
        scaled_output = scaled_bg.filter('scale', 'min(1280, iw)', 'min(720, ih)', force_original_aspect_ratio='decrease')\
                                 .filter('pad', 1280, 720, '(ow-iw)/2', '(oh-ih)/2')
        scaled_output_path = os.path.join(temp_dir, "scaled_bg.mp4")
        
        try:
            ffmpeg.output(scaled_output, scaled_output_path).overwrite_output().run(capture_stdout=True, capture_stderr=True)
            logger.info("背景動画のスケーリングが成功しました。")
        except ffmpeg.Error as e:
            logger.error(f"背景動画のスケーリングに失敗しました: {e.stderr.decode()}")
            raise

        # 2. 画像のオーバーレイを追加
        video_with_images = ffmpeg.input(scaled_output_path)
        for idx, (start, end) in enumerate(image_display_times):
            img_path = os.path.join(temp_dir, f"temp_image_{idx}.png")
            images[idx].save(img_path)
            img_input = ffmpeg.input(img_path)
            video_with_images = video_with_images.overlay(
                img_input,
                x='(main_w-overlay_w)/2',
                y='(main_h-overlay_h)/2',
                enable=f'between(t,{start},{end})'
            )
        
        video_with_images_path = os.path.join(temp_dir, "video_with_images.mp4")
        try:
            ffmpeg.output(video_with_images, video_with_images_path).overwrite_output().run(capture_stdout=True, capture_stderr=True)
            logger.info("画像のオーバーレイが成功しました。")
        except ffmpeg.Error as e:
            logger.error(f"画像のオーバーレイに失敗しました: {e.stderr.decode()}")
            raise

        # 3. テキストの追加
        video_with_text = ffmpeg.input(video_with_images_path)
        for text, start, end in scene_timestamps:
            video_with_text = video_with_text.drawtext(
                text=escape_ffmpeg_text(text),
                fontfile=font_path,
                fontsize=24,
                fontcolor='white',
                box=1,
                boxcolor='black@0.8',
                boxborderw=10,
                x='(w-text_w)/2',
                y='h-text_h-60',
                enable=f'between(t,{start},{end})'
            )
        
        video_with_text_path = os.path.join(temp_dir, "video_with_text.mp4")
        try:
            ffmpeg.output(video_with_text, video_with_text_path).overwrite_output().run(capture_stdout=True, capture_stderr=True)
            logger.info("テキストの追加が成功しました。")
        except ffmpeg.Error as e:
            logger.error(f"テキストの追加に失敗しました: {e.stderr.decode()}")
            raise

        # 4. 音声の結合
        audio_input = ffmpeg.input(mixed_audio_path)
        video_input = ffmpeg.input(video_with_text_path)
        
        output_path = os.path.join(current_dir, f"generated_video_{timestamp}.mp4")
        try:
            ffmpeg.output(video_input, audio_input, output_path, vcodec='libx264', acodec='aac').overwrite_output().run(capture_stdout=True, capture_stderr=True)
            logger.info("音声の結合が成功しました。")
        except ffmpeg.Error as e:
            logger.error(f"音声の結合に失敗しました: {e.stderr.decode()}")
            raise

        logger.info(f"動画が正常に作成されました: {output_path}")
        return output_path

    except Exception as e:
        logger.error(f"動画作成中にエラーが発生しました: {str(e)}")
        raise

    finally:
        # 一時ファイルの削除
        for idx in range(len(image_display_times)):
            img_path = os.path.join(temp_dir, f"temp_image_{idx}.png")
            if not os.path.exists(img_path):
                logger.error(f"一時画像ファイルが存在しません: {img_path}")
                raise FileNotFoundError(f"一時画像ファイルが存在しません: {img_path}")
            else:
                logger.info(f"一時画像ファイルが存在します: {img_path}")
            os.remove(img_path)
        os.remove(combined_audio_path)
        os.remove(mixed_audio_path)

        # 一時ディレクトリの削除
        if os.path.exists(temp_dir):
            import shutil
            shutil.rmtree(temp_dir)

@app.get("/test-script")
async def test_script():
    try:
        script = generate_script("テストノート内容")
        return {"script": script}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def verify_files():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    font_path = os.path.join(current_dir, 'font.ttf')
    movie_dir = os.path.join(current_dir, "movie")
    music_dir = os.path.join(current_dir, "music")
    
    logger.info(f"フォントファイルの存在: {os.path.exists(font_path)} ({font_path})")
    logger.info(f"movieディレクトリの存在: {os.path.exists(movie_dir)} ({movie_dir})")
    logger.info(f"musicディレクトリの存在: {os.path.exists(music_dir)} ({music_dir})")
    
    if not os.path.exists(font_path):
        logger.error("font.ttfが存在しません。")
    
    if not os.path.exists(movie_dir) or not os.listdir(movie_dir):
        logger.error("movieディレクトリにmp4ファイルが存在しないか、ディレクトリ自体が存在しません。")
    
    if not os.path.exists(music_dir) or not os.listdir(music_dir):
        logger.error("musicディレクトリに音楽ファイルが存在しないか、ディレクトリ自体が存在しません。")

async def send_error_to_client(client_id: str, error_message: str):
    # この関数を適切に実装してください。例えば：
    logger.error(f"Error for client {client_id}: {error_message}")
    # クライアントにエラーを送信する実際の処理をここに追加

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))