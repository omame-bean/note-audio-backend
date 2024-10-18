from fastapi import APIRouter
from services.script_generator import generate_script

router = APIRouter()

@router.get("/test-script")
async def test_script():
    script = await generate_script("テストノート内容")
    return {"script": script}

@router.get("/check-ffmpeg")
async def check_ffmpeg():
    import subprocess
    try:
        result = subprocess.run(["ffmpeg", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode == 0:
            ffmpeg_version = result.stdout.split('\n')[0]
            return {"ffmpeg_version": ffmpeg_version}
        else:
            return {"error": "FFmpegが見つかりませんでした。"}
    except Exception as e:
        return {"error": str(e)}
