import tempfile
import os
from google.cloud import texttospeech
from pydub import AudioSegment

# 環境変数からAPIキーを読み込む
api_key = os.environ.get('GOOGLE_CLOUD_API_KEY')

# クライアントの初期化（APIキーを使用）
client = texttospeech.TextToSpeechClient(client_options={"api_key": api_key})

async def generate_audio(script: list) -> tuple:
    audio_clips = []
    durations = []
    for scene in script:
        # 入力テキストを設定
        synthesis_input = texttospeech.SynthesisInput(text=scene['script'])

        # 音声パラメータを設定
        voice = texttospeech.VoiceSelectionParams(
            language_code="ja-JP",  # 日本語に設定（必要に応じて変更してください）
            name="ja-JP-Neural2-B"  # 適切な音声を選択
        )

        # オーディオ設定
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3,
            speaking_rate=1.35,  # スピードを1.3倍に設定
            pitch=2.0  # ピッチを1.15倍に設定（-20.0から20.0の範囲で、1.15倍は約2.9に相当）
        )

        # リクエストを送信
        response = client.synthesize_speech(
            input=synthesis_input, voice=voice, audio_config=audio_config
        )

        # 一時ファイルに音声データを保存
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as temp_audio_file:
            temp_audio_file.write(response.audio_content)
            audio_clips.append(temp_audio_file.name)
        
        duration = get_audio_duration(temp_audio_file.name)
        durations.append(duration)
    
    return audio_clips, durations

def get_audio_duration(audio_path: str) -> float:
    audio = AudioSegment.from_file(audio_path)
    return len(audio) / 1000.0
