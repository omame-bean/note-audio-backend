import tempfile
from openai import OpenAI
from config import OPENAI_API_KEY
from pydub import AudioSegment

client = OpenAI(api_key=OPENAI_API_KEY)

async def generate_audio(script: list) -> tuple:
    audio_clips = []
    durations = []
    for scene in script:
        response = client.audio.speech.create(
            model="tts-1",
            voice="nova",
            input=scene['script']
        )
        
        audio_content = response.content

        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as temp_audio_file:
            temp_audio_file.write(audio_content)
            audio_clips.append(temp_audio_file.name)
        
        duration = get_audio_duration(temp_audio_file.name)
        durations.append(duration)
    
    return audio_clips, durations

def get_audio_duration(audio_path: str) -> float:
    audio = AudioSegment.from_file(audio_path)
    return len(audio) / 1000.0
