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
        以下の内容を表現するリアルなイラストを生成してください。人物はいりません。：
        
        {combined_script}
        
        # Requirements:
        - Please be sure to draw in a realistic illustration style.
        - Please do not draw people. Please draw landscapes and scenery.
        - Please include elements that are directly related to the script content.
        - Please express the atmosphere of the scene, including the background.
        - Please do not include text or letters.
        
        This illustration will be used as a scene in a video. Please ensure that it has an appropriate composition and content.
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
