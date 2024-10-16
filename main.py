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

        background_image = select_background(request.note_content)
        logger.info(f"背景画像が選択されました: {background_image}")

        images = generate_images(script)
        logger.info("画像が正常に生成されました")

        audio = generate_audio(script)
        logger.info("音声が正常に合成されました")

        video_filename = create_video(script, images, audio, background_image)
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
            {"role": "system", "content": "ノート内容から30秒前後の動画用の台詞を生成してください。各シーンを辞書形式で返してください。各シーンは'scene_number'、'description'、'script'のキーを持つ必要があります。返答は必ず有効なJSONフォーマットにしてください。全て日本語で書いてください。"},
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
        combined_script = "\n".join([f"シーン{scene['scene_number']}: {scene['script']}" for scene in group_scenes])
        
        prompt = f"""
        以下の内容を表現する手描きのイラストを生成してください：
        
        {combined_script}
        
        要件：
        - 必ず手描きのイラストスタイルで描いてください。写真や実写は不可。
        - スクリプトの内容に直接関連する要素を含めてください。
        - シンプルで明確な線画で、カラフルに彩色してください。
        - 背景も含め、シーンの雰囲気を表現してください。
        - 人物、物体、環境などスクリプトに出てくる要素を必ず含めてください。
        - マイク録音に関する内容の場合、必ずマイクや録音機器を描いてください。
        
        このイラストは動画のシーンとして使用されます。適切な構図と内容を心がけてください。
        """
        
        logger.debug(f"画像生成プロンプト（グループ {i//5 + 1}）:\n{prompt}")
        
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
        
        logger.debug(f"生成された画像のURL（グループ {i//5 + 1}）: {image_url}")
    
    return images

def generate_audio(script: list) -> list:
    audio_clips = []
    for scene in script:
        response = client.audio.speech.create(
            model="tts-1",
            voice="shimmer",
            input=scene['script']
        )
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as temp_audio_file:
            temp_audio_file.write(response.content)
            audio_clips.append(temp_audio_file.name)

    return audio_clips

def select_background(note_content: str) -> str:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    img_dir = os.path.join(current_dir, "img")
    backgrounds = [f for f in os.listdir(img_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    prompt = f"""
    以下のノート内容に最も適した背景画像を選んでください。
    選択肢は次の通りです: {', '.join(backgrounds)}
    
    ノート内容:
    {note_content}
    
    最も適切な背景画像のファイル名のみを返してください。
    """
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "あなたは与えられたノート内容に基づいて最適な背景画像を選択する専門家です。"},
            {"role": "user", "content": prompt}
        ]
    )
    
    selected_background = response.choices[0].message.content.strip()
    logger.debug(f"選択された背景画像: {selected_background}")
    
    return os.path.join(img_dir, selected_background)

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

def create_video(script: list, images: list, audio_clips: list, background_image: str, orientation='landscape') -> str:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    timestamp = int(time.time())
    temp_output = os.path.join(current_dir, f"temp_generated_video_{timestamp}.mp4")
    output_path = os.path.join(current_dir, f"generated_video_{timestamp}.mp4")
    combined_audio_path = os.path.join(current_dir, f"combined_audio_{timestamp}.mp3")

    logger.debug(f"Current directory: {current_dir}")
    logger.debug(f"Temp output path: {temp_output}")
    logger.debug(f"Combined audio path: {combined_audio_path}")

    frame_size = (1280, 720)
    fps = 24

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_output, fourcc, fps, frame_size)

    combined_audio_path = combine_audio(audio_clips, combined_audio_path)

    total_duration = get_audio_duration(combined_audio_path)
    total_frames = int(total_duration * fps)

    if total_duration <= 0:
        logger.error("オーディオの長さが0秒以下です。")
        raise ValueError("オーディオの長さが0秒以下です。")

    background = Image.open(background_image).convert("RGBA")
    background = background.resize(frame_size, Image.Resampling.LANCZOS)

    scene_durations = [get_audio_duration(clip) for clip in audio_clips]
    scene_frames = [int(duration * fps) for duration in scene_durations]

    current_frame = 0
    for scene_index, scene_frame_count in enumerate(scene_frames):
        image_index = scene_index // 5
        image = images[image_index][1].convert("RGBA")
        image.thumbnail((frame_size[0], frame_size[1]), Image.Resampling.LANCZOS)
        img_width, img_height = image.size

        motion_type = random.choice(['pan', 'zoom_in', 'zoom_in_out'])

        for frame in range(scene_frame_count):
            progress = frame / scene_frame_count
            frame_image = background.copy()

            if motion_type == 'pan':
                offset_x = int((frame_size[0] - img_width) * progress)
                offset_y = (frame_size[1] - img_height) // 2
                frame_image.alpha_composite(image, (offset_x, offset_y))
            elif motion_type == 'zoom_in':
                zoom_factor = 1 + (0.5 * progress)  # ここを調整
                zoomed_size = (int(img_width * zoom_factor), int(img_height * zoom_factor))
                zoomed_image = image.resize(zoomed_size, Image.Resampling.LANCZOS)
                offset_x = (frame_size[0] - zoomed_size[0]) // 2
                offset_y = (frame_size[1] - zoomed_size[1]) // 2
                frame_image.alpha_composite(zoomed_image, (offset_x, offset_y))
            else:  # zoom_in_out
                if progress < 0.5:
                    zoom_factor = 1 + (0.5 * (progress * 2))  # ここを調整
                else:
                    zoom_factor = 1.5 - (0.5 * ((progress - 0.5) * 2))  # ここを調整
                zoomed_size = (int(img_width * zoom_factor), int(img_height * zoom_factor))
                zoomed_image = image.resize(zoomed_size, Image.Resampling.LANCZOS)
                offset_x = (frame_size[0] - zoomed_size[0]) // 2
                offset_y = (frame_size[1] - zoomed_size[1]) // 2
                frame_image.alpha_composite(zoomed_image, (offset_x, offset_y))

            frame = cv2.cvtColor(np.array(frame_image), cv2.COLOR_RGBA2BGR)
            out.write(frame)
            current_frame += 1

    out.release()
    logger.debug(f"一時動画ファイルが作成されました: {temp_output}")

    if not os.path.exists(temp_output):
        logger.error(f"一時動画ファイルが見つかりません: {temp_output}")
        raise FileNotFoundError(f"一時動画ファイルが見つかりません: {temp_output}")

    if not os.path.exists(combined_audio_path):
        logger.error(f"結合された音声ファイルが見つかりません: {combined_audio_path}")
        raise FileNotFoundError(f"結合された音声ファイルが見つかりません: {combined_audio_path}")

    try:
        input_video = ffmpeg.input(temp_output)
        input_audio = ffmpeg.input(combined_audio_path)

        output = (
            ffmpeg
            .output(
                input_video,
                input_audio,
                output_path,
                vcodec='libx264',
                preset='ultrafast',  # エンコード速度を最優先
                crf=23,  # 画質と圧縮率のバランスを調整（値が小さいほど高画質）
                acodec='aac',
                audio_bitrate='128k',  # 音声ビットレートを調整
                strict='experimental'
            )
            .global_args('-map', '0:v:0', '-map', '1:a:0')
            .overwrite_output()
        )

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
