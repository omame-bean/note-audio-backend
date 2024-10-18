import asyncio
import json

client_progress = {}

async def progress_generator(client_id: str):
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
                yield f"data: {step}\n\n"
                last_sent_index += 1
            
            if any(json.loads(step).get("step") == "最終出力" for step in new_steps):
                break
            
            await asyncio.sleep(1)
        except Exception as e:
            await asyncio.sleep(1)
            retry_count += 1
    
    if retry_count >= max_retries:
        yield f"data: {json.dumps({'error': '最大再試行回数に達しました。'})}\n\n"

async def update_progress(client_id: str, step: str, status: str, message: str = None, video_url: str = None):
    progress = {
        "step": step,
        "status": status,
        "message": message,
        "video_url": video_url
    }
    if client_id not in client_progress:
        client_progress[client_id] = []
    client_progress[client_id].append(json.dumps(progress))
