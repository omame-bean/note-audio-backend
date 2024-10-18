import json
import re
from openai import OpenAI
from config import OPENAI_API_KEY

client = OpenAI(api_key=OPENAI_API_KEY)

async def generate_script(note_content: str) -> list:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "ノート内容から30秒前後の動画用の台詞を成してください。各シーンを辞書形式で返してください。各シーンは'scene_number'、'description'、'script'のキーを持つ必要があります。返答は必ず有効なJSONフォーマットにしてください。全て日本語でいてください。"},
            {"role": "user", "content": note_content}
        ]
    )
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
