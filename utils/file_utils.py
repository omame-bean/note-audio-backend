import os
import textwrap
from openai import OpenAI
from config import OPENAI_API_KEY

client = OpenAI(api_key=OPENAI_API_KEY)

def select_background(note_content: str) -> str:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    movie_dir = os.path.join(current_dir, "..", "movie")
    
    backgrounds = [f for f in os.listdir(movie_dir) if f.endswith('.mp4')]
    
    if not backgrounds:
        raise FileNotFoundError("背景動画が見つかりません。")
    
    prompt = f"""
    以下のノート内容に最も適した背景動画を選んでください。
    選択肢は次の通りです: {', '.join(backgrounds)}
    
    ノート内容:
    {note_content}
    
    最も適切な背景動画のファイル名のみを返してください。
    """
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "あなたは与えられたノート内容に基づいて最適な背景動画を選択する専門家です。"},
            {"role": "user", "content": prompt}
        ]
    )
    
    selected_background = response.choices[0].message.content.strip()
    
    if selected_background not in backgrounds:
        selected_background = backgrounds[0]
    
    full_path = os.path.join(movie_dir, selected_background)
    
    return full_path

def select_background_music(note_content: str) -> str:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    music_dir = os.path.join(current_dir, "..", "music")
    
    music_files = [f for f in os.listdir(music_dir) if f.endswith(('.mp3', '.wav'))]
    
    if not music_files:
        raise FileNotFoundError("背景音楽が見つかりません。")
    
    prompt = f"""
    以下のノート内容に最も適した背景音楽を選んでください。
    選択肢は次の通りです: {', '.join(music_files)}
    
    ノート内容:
    {note_content}
    
    最も適切な背景音楽のファイル名のみを返してください。
    """
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "あなたは与えられたノート内容に基づいて最適な背景音楽を選択する専門家です。"},
            {"role": "user", "content": prompt}
        ]
    )
    
    selected_music = response.choices[0].message.content.strip()
    
    if selected_music not in music_files:
        selected_music = music_files[0]
    
    full_path = os.path.join(music_dir, selected_music)
    
    return full_path

def wrap_text(text: str, max_width: int) -> str:
    words = text.split()
    lines = []
    current_line = []
    current_width = 0

    for word in words:
        if current_width + len(word) + 1 > max_width:
            lines.append(' '.join(current_line))
            current_line = [word]
            current_width = len(word)
        else:
            current_line.append(word)
            current_width += len(word) + 1

    if current_line:
        lines.append(' '.join(current_line))

    return '\n'.join(lines)

def escape_ffmpeg_text(text: str) -> str:
    text = text.replace('\\', '\\\\').replace("'", "\\'")
    text = text.replace(',', '\\,').replace(':', '\\:')
    return text
