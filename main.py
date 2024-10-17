from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
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
from fastapi.responses import FileResponse
from pydub import AudioSegment
import random
import textwrap  # 追加

# ロギングの基本設定
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

from dotenv import load_dotenv
load_dotenv()

import subprocess

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
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://your-frontend-domain.com"],  # フロントエンドのURLを指定
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# OpenAIクライアントの初期化
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class VideoRequest(BaseModel):
    note_content: str

class VideoResponse(BaseModel):
    video_url: str

@app.get("/download-video/{video_filename}")
async def download_video(video_filename: str):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    video_path = os.path.join(current_dir, video_filename)
    logger.debug(f"Requested video path: {video_path}")
    logger.debug(f"File exists: {os.path.exists(video_path)}")
    if os.path.exists(video_path):
        return FileResponse(
            video_path,
            media_type="video/mp4",
            filename=video_filename,
            headers={"Content-Disposition": f'attachment; filename="{video_filename}"'}
        )
    else:
        raise HTTPException(status_code=404, detail=f"動画が見つかりません: {video_path}")

@app.post("/generate-video", response_model=VideoResponse)
async def generate_video(request: VideoRequest):
    try:
        script = generate_script(request.note_content)
        logger.info("台詞が正常に生成されました")

        background_video = select_background(request.note_content)
        logger.info(f"選択された背景動画: {background_video}")
        
        if not os.path.exists(background_video):
            raise FileNotFoundError(f"選択された背景動画が見つかりません: {background_video}")

        images = generate_images(script)
        logger.info("画像が正常に生成されました")

        audio_clips, durations = generate_audio(script)
        logger.info("音声が正常に合成されました")

        # 3シーンごとのグループ分けと表示時間の計算
        image_display_times = []
        cumulative_time = 0.0
        for i in range(0, len(durations), 3):
            group_durations = durations[i:i+3]
            start_time = cumulative_time
            display_duration = sum(group_durations)
            end_time = start_time + display_duration
            image_display_times.append((images[i//3][1], start_time, end_time))
            cumulative_time = end_time

        video_filename = create_video(script, images, audio_clips, background_video, image_display_times)
        logger.info(f"動画が正確に作成されました: {video_filename}")
        return VideoResponse(video_url=f"/download-video/{os.path.basename(video_filename)}")
    except Exception as e:
        logger.error(f"generate_videoで予期せぬエラーが発生しました: {str(e)}")
        logger.error(f"スタックトレース: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"内部サーバーエラーが発生しました: {str(e)}")

def generate_script(note_content: str) -> list:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "ノート内容から30秒前後の動画用の台詞を成してください。各シーンを辞書形式で返してください。各シーンは'scene_number'、'description'、'script'のキーを持つ必要があります。返答は必ず有効なJSONフォーマットにしてください。全て日本語で書いてください。"},
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

def generate_images(script: list) -> list:
    images = []
    for i in range(0, len(script), 3):  # 5から3に変更
        group_scenes = script[i:i+3]  # 5から3に変更
        combined_script = "\n".join([f"シーン{scene['scene_number']}: {scene['script']}" for scene in group_scenes])
        
        prompt = f"""
        以下の内容を表現する手描きのイラスト生成してください：
        
        {combined_script}
        
        要件：
        - 必ず手描きのイラストスタイルで描いてください。写真や実写は不可。
        - スクリプトの内容に直接関連する要素を含めてください。
        - シンプルで明確な線画で、カラフルに彩色してください。
        - 背景も含め、シーンの雰囲気を表現してください。
        - 画像の中に文字は絶対に入れないでください。
        
        このイラストは動画のシーンとして使用されます。適切な構図と内容を心がけてください。
        """
        
        logger.debug(f"画像生成プロンプト（グループ {i//3 + 1}）:\n{prompt}")  # 5から3に変更
        
        response = client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            n=1,
            size="1024x1024"
        )
        image_url = response.data[0].url
        image_response = requests.get(image_url)
        img = Image.open(BytesIO(image_response.content))
        images.append((i, img))
        
        logger.debug(f"生された画像のURL（グループ {i//3 + 1}）: {image_url}")  # 5から3に変更
    
    return images

def generate_audio(script: list) -> tuple:
    audio_clips = []
    durations = []
    for scene in script:
        response = client.audio.speech.create(
            model="tts-1",
            voice="shimmer",
            input=scene['script']
        )
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as temp_audio_file:
            temp_audio_file.write(response.content)
            audio_clips.append(temp_audio_file.name)
        
        duration = get_audio_duration(temp_audio_file.name)
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

def create_video(script: list, images: list, audio_clips: list, background_video: str, image_display_times: list, orientation='landscape') -> str:
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
    logger.debug(f"Background video: {background_video}")

    combined_audio_path = combine_audio(audio_clips, combined_audio_path)
    total_duration = get_audio_duration(combined_audio_path)

    if total_duration <= 0:
        logger.error("オーディオの長さが0秒以下です。")
        raise ValueError("オーディオの長さが0秒以下です。")

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

    try:
        # 背景動画を無限ループさせて、音声の長さに合わせる
        bg_input = ffmpeg.input(background_video, stream_loop=-1, t=total_duration)
        video = bg_input.video

        # 背景動画をスケールとパッドを適用
        video = video.filter('scale', 'min(1280, iw)', 'min(720, ih)', force_original_aspect_ratio='decrease')\
                     .filter('pad', 1280, 720, '(ow-iw)/2', '(oh-ih)/2')

        # 画像をオーバーレイ
        for idx, (img, start, end) in enumerate(image_display_times):
            img_path = f"temp_image_{idx}.png"
            img.save(img_path)
            img_input = ffmpeg.input(img_path)
            
            # 画像を中央にオーバーレイ
            video = video.overlay(
                img_input,
                x='(main_w-overlay_w)/2',
                y='(main_h-overlay_h)/2',
                enable=f'between(t,{start},{end})'
            )

        # フォントファイルの絶対パスを取得
        font_path = os.path.join(current_dir, 'font.ttf')
        font_path = os.path.abspath(font_path)  # 絶対パスに変換

        # フォントファイルの存在確認
        if not os.path.exists(font_path):
            logger.error(f"フォントファイルが見つかりません: {font_path}")
            raise FileNotFoundError(f"フォントファイルが見つかりません: {font_path}")
        
        logger.debug(f"Using font file: {font_path}")

        # テロップを各シーンごとに追加
        for idx, (text, start, end) in enumerate(scene_timestamps):
            logger.debug(f"Adding text: '{text}' from {start} to {end}")
            video = video.drawtext(
                text=text,
                fontfile=font_path,
                fontsize=24,
                fontcolor='white',
                box=1,
                boxcolor='black@0.8',
                boxborderw=10,
                x='(w-text_w)/2',  # テキストを中央配置
                y='h-text_h-60',    # 下から60pxの位置に配置
                enable=f'between(t,{start},{end})',
                line_spacing=5
                # reload=1  # 削除しました
            )

        audio_input = ffmpeg.input(combined_audio_path)

        # 動画と音声を結合して出力
        output = (
            ffmpeg
            .output(video, audio_input, output_path, vcodec='libx264', acodec='aac', audio_bitrate='128k', t=total_duration)
            .overwrite_output()
        )
        
        logger.info("FFmpegコマンドを実行します")
        logger.info(f"FFmpegコマンド: {' '.join(ffmpeg.compile(output))}")
        
        ffmpeg.run(output)
        logger.info(f"動画が正常に作成されました: {output_path}")

        # 一時ファイルの削除
        for idx in range(len(image_display_times)):
            os.remove(f"temp_image_{idx}.png")
        os.remove(combined_audio_path)

    except ffmpeg.Error as e:
        stderr_output = e.stderr.decode() if e.stderr else 'No stderr'
        logger.error(f"FFmpeg error: {stderr_output}")
        logger.error(f"FFmpeg error details: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error in create_video: {str(e)}")
        raise

    return output_path

@app.get("/test-script")
async def test_script():
    try:
        script = generate_script("テストノート内容")
        return {"script": script}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
