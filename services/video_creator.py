import os
import time
import ffmpeg
from pydub import AudioSegment
from utils.file_utils import escape_ffmpeg_text
import asyncio
import logging
import tempfile
import subprocess
from pydub.utils import mediainfo
from PIL import Image
import textwrap

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

async def create_video(script: list, images: list, audio_clips: list, background_video: str, background_music: str, durations: list, video_type: str = 'landscape') -> str:
    try:
        logger.info("動画作成開始")
        current_dir = os.path.dirname(os.path.abspath(__file__))
        timestamp = int(time.time())
        output_path = os.path.join(current_dir, "..", f"generated_video_{timestamp}.mp4")
        combined_audio_path = os.path.join(current_dir, "..", f"combined_audio_{timestamp}.mp3")

        logger.info("音声ファイルの結合開始")
        combined_audio_path = await asyncio.to_thread(combine_audio, audio_clips, combined_audio_path)
        total_duration = sum(durations)
        logger.info(f"音声ファイルの結合完了。総再生時間: {total_duration}秒")

        logger.info("背景音楽の準備開")
        background_music_audio = await asyncio.to_thread(AudioSegment.from_file, background_music)
        background_music_audio = background_music_audio - 20  # 音量を30%下げる（-10dB）
        if len(background_music_audio) < total_duration * 1000:
            background_music_audio = background_music_audio * (int(total_duration * 1000 / len(background_music_audio)) + 1)
        background_music_audio = background_music_audio[:int(total_duration * 1000)]
        logger.info("背景音楽の準備完了")
        
        logger.info("音声ミキシング開始")
        voice_audio = await asyncio.to_thread(AudioSegment.from_file, combined_audio_path)
        voice_audio = voice_audio + 0.1  # 音量
        mixed_audio = voice_audio.overlay(background_music_audio)
        mixed_audio_path = os.path.join(current_dir, "..", f"mixed_audio_{timestamp}.mp3")
        await asyncio.to_thread(mixed_audio.export, mixed_audio_path, format='mp3')
        logger.info("音声ミキシング完了")

        logger.info("テロップ用タイムスタンプの作成開始")
        scene_timestamps = []
        cumulative_time = 0.0
        for idx, scene in enumerate(script):
            scene_duration = durations[idx] if idx < len(durations) else 5.0  # デフォルトを5秒とする
            scene_start = cumulative_time
            scene_end = cumulative_time + scene_duration
            # wrap_text の代わりに textwrap.wrap を使用
            wrapped_text = '\n'.join(textwrap.wrap(scene['script'], width=40))
            scene_timestamps.append((wrapped_text, scene_start, scene_end))
            cumulative_time = scene_end
        logger.info("テロップ用タイムスタンプの作成完了")

        logger.info("FFmpeg処理開始")
        if video_type == 'portrait':
            width, height = 720, 1280
        else:
            width, height = 1280, 720

        bg_input = ffmpeg.input(background_video, stream_loop=-1, t=total_duration)
        video = bg_input.video

        # ここを修正
        video = (
            video.filter('scale', w=width, h=height, force_original_aspect_ratio='increase')
            .filter('crop', w=width, h=height)
            .filter('setsar', '1')
        )

        image_display_times = calculate_image_display_times(durations)
        temp_dir = tempfile.mkdtemp()
        try:
            for idx, (start, end) in enumerate(image_display_times):
                img_path = os.path.join(temp_dir, f"temp_image_{idx}.png")
                resized_image = resize_image(images[idx], width, height)
                await asyncio.to_thread(resized_image.save, img_path)
                img_input = ffmpeg.input(img_path)
                
                video = video.overlay(
                    img_input,
                    x='0',
                    y='0',
                    enable=f'between(t,{start},{end})'
                )

            font_path = os.path.join(current_dir, '..', 'font.ttf')
            font_path = os.path.abspath(font_path)

            for idx, (text, start, end) in enumerate(scene_timestamps):
                if video_type == 'portrait':
                    y_position = 'h-th-120'  # テキストの高さを考慮
                    fontsize = 26
                    max_width = 30  # ポートレートモードでの最大文字数
                else:
                    y_position = 'h-th-60'  # テキストの高さを考慮
                    fontsize = 26
                    max_width = 40  # ランドスケープモードでの最大文字数

                # テキストを折り返す
                wrapped_lines = textwrap.wrap(text, width=max_width)
                wrapped_text = '\n'.join(wrapped_lines)
                escaped_text = escape_ffmpeg_text(wrapped_text)

                video = video.drawtext(
                    text=escaped_text,
                    fontfile=font_path,
                    fontsize=fontsize,
                    fontcolor='white',
                    box=1,
                    boxcolor='black@0.8',
                    boxborderw=10,
                    x='(w-tw)/2',  # 水平方向の中央配置
                    y=y_position,
                    enable=f'between(t,{start},{end})',
                    line_spacing=5
                )

            # フェード処理の調整
            video_fade_duration = 2.0  # ビデオのフェード時間
            audio_fade_duration = 0.5  # 音声のフェード時間

            video = video.filter('fade', type='in', duration=video_fade_duration)
            video = video.filter('fade', type='out', start_time=total_duration-video_fade_duration, duration=video_fade_duration)

            audio_input = ffmpeg.input(mixed_audio_path)

            # 音声のフェードイン・フェードアウトを調整
            audio_input = audio_input.filter('afade', type='in', duration=audio_fade_duration)
            audio_input = audio_input.filter('afade', type='out', start_time=total_duration-audio_fade_duration, duration=audio_fade_duration)

            audio_input = (
                audio_input
                .filter('afftdn', nr=2, nt='w', om='o')  # ノイズ除去フィルター
            )

            output = (
                ffmpeg
                .output(video, audio_input, output_path, vcodec='libx264', acodec='aac', audio_bitrate='192k', t=total_duration)
                .overwrite_output()
            )
            
            output = output.global_args('-threads', '2')
            logger.info("FFmpeg処理実行開始")
            
            # FFmpegコマンドを文字列として取得
            ffmpeg_cmd = ffmpeg.compile(output)
            logger.info(f"FFmpegコマンド: {' '.join(ffmpeg_cmd)}")

            # サブプロセスとしてFFmpegを実行
            process = await asyncio.create_subprocess_exec(
                *ffmpeg_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            # 標準出力と標準エラー出力を取得
            stdout, stderr = await process.communicate()

            if process.returncode != 0:
                logger.error(f"FFmpegエラー: {stderr.decode()}")
                raise RuntimeError(f"FFmpegエラー: {stderr.decode()}")

            logger.info("FFmpeg処理完了")

        finally:
            # 一時ファイルの削除
            for idx in range(len(image_display_times)):
                img_path = os.path.join(temp_dir, f"temp_image_{idx}.png")
                if os.path.exists(img_path):
                    os.remove(img_path)
            os.rmdir(temp_dir)
            os.remove(combined_audio_path)
            os.remove(mixed_audio_path)
            logger.info("一時ファイルの削除完了")

        logger.info(f"動画作成完了: {output_path}")
        return output_path
    except Exception as e:
        logger.error(f"動画作成中にエラーが発生しました: {str(e)}", exc_info=True)
        raise

def combine_audio(audio_paths: list, output_path: str) -> str:
    combined = AudioSegment.empty()
    target_sample_rate = 44100  # 目標サンプリングレート

    for audio_path in audio_paths:
        audio = AudioSegment.from_file(audio_path)
        info = mediainfo(audio_path)
        original_sample_rate = int(info['sample_rate'])
        
        if original_sample_rate != target_sample_rate:
            audio = audio.set_frame_rate(target_sample_rate)
        
        combined += audio

    combined.export(output_path, format='mp3', parameters=["-ar", str(target_sample_rate)])
    return output_path

def calculate_image_display_times(durations):
    image_display_times = []
    cumulative_time = 0.0
    for i in range(0, len(durations), 3):
        group_durations = durations[i:i+3]
        start_time = cumulative_time
        display_duration = sum(group_durations)
        end_time = start_time + display_duration
        image_display_times.append((start_time, end_time))
        cumulative_time = end_time
    return image_display_times

def normalize_audio(audio_segment: AudioSegment) -> AudioSegment:
    return audio_segment.normalize()

def resize_image(image: Image.Image, width: int, height: int) -> Image.Image:
    aspect_ratio = image.width / image.height
    target_ratio = width / height

    # 画像の新しいサイズを計算（高さを100px小さくする）
    new_height = height - 100
    new_width = int(new_height * aspect_ratio)

    # 新しい幅が全体の幅を超える場合は、幅に合わせて調整
    if new_width > width:
        new_width = width
        new_height = int(width / aspect_ratio)

    resized_image = image.resize((new_width, new_height), Image.LANCZOS)

    # 新しい背景画像を作成（透明）
    background = Image.new('RGBA', (width, height), (0, 0, 0, 0))

    # リサイズした画像を中央に配置
    paste_x = (width - new_width) // 2
    paste_y = (height - new_height) // 2
    background.paste(resized_image, (paste_x, paste_y))

    return background
