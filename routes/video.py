from fastapi import APIRouter, HTTPException, BackgroundTasks, Query
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
import os
import json
import asyncio
from services.script_generator import generate_script
from services.audio_generator import generate_audio
from services.image_generator import generate_images
from services.video_creator import create_video
from utils.progress_utils import progress_generator, update_progress
from utils.file_utils import select_background, select_background_music
import logging
import traceback

logger = logging.getLogger(__name__)

router = APIRouter()

class VideoRequest(BaseModel):
    client_id: str
    note_content: str
    video_type: str = 'landscape'  # 'landscape' または 'portrait'

@router.post("/generate-video")
async def generate_video(request: VideoRequest, background_tasks: BackgroundTasks):
    try:
        # note_contentの検証
        if not request.note_content:
            raise HTTPException(status_code=400, detail="ノート内容が指定されていません")

        # ビデオタイプの検証
        if request.video_type not in ['landscape', 'portrait']:
            raise HTTPException(status_code=400, detail="無効なビデオタイプです。'landscape'または'portrait'を指定してください")

        # バックグラウンドタスクとして動画生成を実行
        background_tasks.add_task(run_video_generation, request.client_id, request.note_content, request.video_type)

        return {"message": "動画生成タスクが開始されました", "client_id": request.client_id}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/events/{client_id}")
async def events(client_id: str):
    return StreamingResponse(progress_generator(client_id), media_type="text/event-stream")

@router.get("/download-video/{video_filename}")
async def download_video(video_filename: str, client_id: str = Query(...)):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    video_path = os.path.join(current_dir, "..", video_filename)
    if os.path.exists(video_path):
        return FileResponse(video_path, media_type="video/mp4", filename=video_filename)
    else:
        raise HTTPException(status_code=404, detail=f"動画が見つかりません: {video_path}")

async def run_video_generation(client_id: str, note_content: str, video_type: str):
    try:
        logger.info(f"動画生成開始 - クライアントID: {client_id}, ビデオタイプ: {video_type}")
        await update_progress(client_id, "テキスト解析", "in-progress", "テキスト解析中...")
        script = await generate_script(note_content)
        await update_progress(client_id, "テキスト解析", "completed", "台詞が正常に生成されました")

        await update_progress(client_id, "背景選択", "in-progress", "背景動画を選択中...")
        background_video = await asyncio.to_thread(select_background, note_content)
        await update_progress(client_id, "背景選択", "completed", f"選択された背景動画: {background_video}")

        await update_progress(client_id, "背景音楽選択", "in-progress", "背景音楽を選択中...")
        background_music = await asyncio.to_thread(select_background_music, note_content)
        await update_progress(client_id, "背景音楽選択", "completed", f"選択された背景音楽: {background_music}")

        await update_progress(client_id, "画像生成と音声合成", "in-progress", "画像生成と音声合成を開始...")
        images = await generate_images(script)
        audio_clips, durations = await generate_audio(script)
        await update_progress(client_id, "画像生成と音声合成", "completed", "画像生成と音声合成が完了しました")

        await update_progress(client_id, "動画編集", "in-progress", "動画編集中...")
        logger.info("動画編集開始")
        video_filename = await create_video(script, images, audio_clips, background_video, background_music, durations, video_type)
        logger.info(f"動画編集完了: {video_filename}")
        await update_progress(client_id, "動画編集", "completed", f"動画が正確に作成されました: {video_filename}")

        video_url = f"/download-video/{os.path.basename(video_filename)}"
        await update_progress(client_id, "最終出力", "completed", "動画生成が完了しました", video_url)
        logger.info(f"動画生成完了 - クライアントID: {client_id}")

    except Exception as e:
        error_message = f"動画生成中にエラーが発生しました: {str(e)}"
        logger.error(f"{error_message} - クライアントID: {client_id}")
        logger.error(traceback.format_exc())
        
        # エラーの詳細情報を取得
        if isinstance(e, RuntimeError) and "FFmpegエラー" in str(e):
            ffmpeg_error = str(e)
            logger.error(f"FFmpegエラーの詳細: {ffmpeg_error}")
            error_message += f"\nFFmpegエラーの詳細: {ffmpeg_error}"
        
        await update_progress(client_id, "エラー", "error", error_message)
