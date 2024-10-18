from PIL import Image
from io import BytesIO
import aiohttp
from openai import OpenAI
from config import OPENAI_API_KEY

client = OpenAI(api_key=OPENAI_API_KEY)

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
        
        response = client.images.generate(
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

    return images
