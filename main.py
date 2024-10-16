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
    video_path = os.path.abspath(video_filename)
    logger.debug(f"Requested video path: {video_path}")
    logger.debug(f"File exists: {os.path.exists(video_path)}")
    if os.path.exists(video_path):
        return FileResponse(video_path, media_type="video/mp4", filename=os.path.basename(video_filename))
    else:
        raise HTTPException(status_code=404, detail=f"動画が見つかりません: {video_path}")

# generate_video 関数内で、動画のファイル名を返すように修正
@app.post("/generate-video", response_model=VideoResponse)
async def generate_video(request: VideoRequest):
    try:
        # 1. GPT-4を使用して台詞を生成
        script = generate_script(request.note_content)
        logger.info("台詞が正常に生成されました")
        logger.info(f"生成されたスクリプト: {json.dumps(script, indent=2, ensure_ascii=False)}")

        # 2. 画像を生成
        images = generate_images(script)
        logger.info("画像が正常に生成されました")
        logger.info(f"生成された画像の数: {len(images)}")

        # 3. 音声を合成
        logger.info("音声合成を開始します")
        audio = generate_audio(script)
        logger.info("音声が正常に合成されました")
        logger.info(f"生成された音声クリップの数: {len(audio)}")

        # 4. 動画を作成
        logger.info("動画作成を開始します")
        video_filename = create_video(script, images, audio)
        logger.info(f"動画が正確に作成されました: {video_filename}")
        return VideoResponse(video_url=f"/download-video/{os.path.basename(video_filename)}")
    except Exception as e:
        logger.error(f"generate_videoで予期せぬエラーが発生しました: {str(e)}")
        logger.error(f"エラーの詳細: {type(e).__name__}, {str(e)}")
        logger.error(f"スタックトレース: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"内部サーバーエラーが発生しまし��: {str(e)}")


def generate_script(note_content: str) -> list:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "ノート内容から1〜3分の動画用の台詞を生成してください。各シーンを辞書形式で返してください。各シーンは'scene_number'、'description'、'script'のキーを持つ必要があります。返答は必ず有効なJSONフォーマットにしてください。全て日本語で書いてください。"},
            {"role": "user", "content": note_content}
        ]
    )
    try:
        content = response.choices[0].message.content
        # 余分な文字を削除し、JSONのみを抽出
        content = re.search(r'\{.*\}', content, re.DOTALL)
        if content:
            content = content.group()
        else:
            raise ValueError("有効なJSONが見つかりませんでした。")
        
        parsed_content = json.loads(content)
        
        # 期待する形式に変換
        if "scenes" in parsed_content:
            scenes = parsed_content["scenes"]
        else:
            scenes = parsed_content
        
        # 各シーンが必要なキーを持っているか確認し、必要に応じて追加
        for scene in scenes:
            if "scene_number" not in scene:
                scene["scene_number"] = scenes.index(scene) + 1
            if "description" not in scene:
                scene["description"] = f"シーン {scene['scene_number']}"
            if "script" not in scene and "dialogue" in scene:
                scene["script"] = scene.pop("dialogue")
        
        return scenes
    except json.JSONDecodeError as e:
        print(f"JSONデコードエラー: {e}")
        print(f"APIレスポンス: {content}")
        raise ValueError("APIからの応答を解析できませんでした。")
    except Exception as e:
        print(f"予期せぬエラー: {e}")
        print(f"APIレスポンス: {content}")
        raise ValueError(f"APIからの応答の処理中にエラーが発生しました: {str(e)}")

def generate_images(script: list) -> list:
    images = []
    for i in range(0, len(script), 5):
        group_scenes = script[i:i+5]
        # 各グループのスクリプトを結合
        combined_script = "\n".join([scene['script'] for scene in group_scenes])
        
        response = client.images.generate(
            prompt=combined_script,
            n=1,
            size="1024x1024"
        )
        image_url = response.data[0].url
        image_response = requests.get(image_url)
        img = Image.open(BytesIO(image_response.content))
        images.append(img)
    return images

def generate_audio(script: list) -> list:
    audio_clips = []
    for scene in script:
        response = client.audio.speech.create(
            model="tts-1",
            voice="shimmer",
            input=scene['script']
        )
        
        # 一時ファイルを作成して音声データを書き込む
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as temp_audio_file:
            temp_audio_file.write(response.content)
            audio_clips.append(temp_audio_file.name)

    return audio_clips

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
    return len(audio) / 1000.0  # 秒単位

def wrap_text(text, font, max_width):
    lines = []
    words = text.split()
    current_line = words[0]
    for word in words[1:]:
        test_line = current_line + " " + word
        bbox = font.getbbox(test_line)
        if bbox[2] - bbox[0] <= max_width:
            current_line = test_line
        else:
            lines.append(current_line)
            current_line = word
    lines.append(current_line)
    return lines

def create_video(script: list, images: list, audio_clips: list, orientation='landscape') -> str:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    timestamp = int(time.time())
    temp_output = os.path.join(current_dir, f"temp_generated_video_{timestamp}.mp4")
    output_path = os.path.join(current_dir, f"generated_video_{timestamp}.mp4")
    combined_audio_path = os.path.join(current_dir, f"combined_audio_{timestamp}.mp3")

    logger.debug(f"Current directory: {current_dir}")
    logger.debug(f"Temp output path: {temp_output}")
    logger.debug(f"Combined audio path: {combined_audio_path}")

    frame_size = (1920, 1080)  # 横動画サイズ
    fps = 24

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_output, fourcc, fps, frame_size)

    # オーディオファイルを結合
    combined_audio_path = combine_audio(audio_clips, combined_audio_path)

    # オーディオの長さを取得
    duration = get_audio_duration(combined_audio_path)
    num_frames = int(duration * fps)

    if duration <= 0:
        logger.error("オーディオの長さが0秒以下です。")
        raise ValueError("オーディオの長さが0秒以下です。")

    # 背景を作成（黒色）
    background = Image.new('RGB', frame_size, (0, 0, 0))

    if images:
        group_image = images[0]
        if group_image.size != frame_size:
            group_image = group_image.resize(frame_size, Image.Resampling.LANCZOS)
            logger.debug("Resized image to frame size.")
        offset = ((frame_size[0] - group_image.width) // 2, (frame_size[1] - group_image.height) // 2)
        background.paste(group_image, offset)
        logger.debug("Pasted image onto background.")

    # PIL Image  numpy array に変換
    frame = cv2.cvtColor(np.array(background), cv2.COLOR_RGB2BGR)

    # フレームを書き込む
    for i in range(num_frames):
        out.write(frame)
        if i % 100 == 0:
            logger.debug(f"Written {i} frames.")

    out.release()
    logger.debug(f"一時動画ファイルが作成されました: {temp_output}")
    logger.debug(f"一時動画ファイルのサイズ: {os.path.getsize(temp_output)} bytes")

    # ファイルの存在を確認
    if not os.path.exists(temp_output):
        logger.error(f"一時動画ファイルが見つかりません: {temp_output}")
        raise FileNotFoundError(f"一時動画ファイルが見つかりません: {temp_output}")

    if not os.path.exists(combined_audio_path):
        logger.error(f"結合された音声ファイルが見つかりません: {combined_audio_path}")
        raise FileNotFoundError(f"結合��れた音声ファイルが見つかりません: {combined_audio_path}")

    try:
        # 動画と音声を結合
        input_video = ffmpeg.input(temp_output)
        input_audio = ffmpeg.input(combined_audio_path)

        # ストリームのマッピングを global_args で追加
        output = (
            ffmpeg
            .output(
                input_video,
                input_audio,
                output_path,
                vcodec='libx264',
                acodec='aac',
                strict='experimental'
            )
            .global_args('-map', '0:v:0', '-map', '1:a:0')
            .overwrite_output()
        )

        # FFmpegコマンドをログに出力
        ffmpeg_command = ' '.join(ffmpeg.compile(output))
        logger.debug(f"FFmpeg command: {ffmpeg_command}")

        ffmpeg.run(output, capture_stdout=True, capture_stderr=True)
        logger.info(f"動画が正常に作成されました: {output_path}")
    except ffmpeg.Error as e:
        logger.error(f"FFmpeg error: {e.stderr.decode() if e.stderr else 'No stderr'}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error in create_video: {str(e)}")
        raise

    # 一時ファイルの削除
    if os.path.exists(temp_output):
        os.remove(temp_output)
    if os.path.exists(combined_audio_path):
        os.remove(combined_audio_path)

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
