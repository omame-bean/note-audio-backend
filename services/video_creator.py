import os
import time
import ffmpeg
from pydub import AudioSegment
from utils.file_utils import wrap_text, escape_ffmpeg_text
import asyncio
import logging
import tempfile
import subprocess
from pydub.utils import mediainfo

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

async def create_video(script: list, images: list, audio_clips: list, background_video: str, background_music: str, durations: list) -> str:
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
        background_music_audio = background_music_audio - 22  # 音量を30%下げる（-10dB）
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
            wrapped_text = wrap_text(scene['script'], max_width=40)  # テキストを折り返す
            scene_timestamps.append((wrapped_text, scene_start, scene_end))
            cumulative_time = scene_end
        logger.info("テロップ用タイムスタンプの作成完了")

        logger.info("FFmpeg処理開始")
        bg_input = ffmpeg.input(background_video, stream_loop=-1, t=total_duration)
        video = bg_input.video

        video = video.filter('scale', 'min(1280, iw)', 'min(720, ih)', force_original_aspect_ratio='decrease')\
                     .filter('pad', 1280, 720, '(ow-iw)/2', '(oh-ih)/2')

        image_display_times = calculate_image_display_times(durations)
        temp_dir = tempfile.mkdtemp()
        try:
            for idx, (start, end) in enumerate(image_display_times):
                img_path = os.path.join(temp_dir, f"temp_image_{idx}.png")
                await asyncio.to_thread(images[idx].save, img_path)
                img_input = ffmpeg.input(img_path)
                
                video = video.overlay(
                    img_input,
                    x='(main_w-overlay_w)/2',
                    y='(main_h-overlay_h)/2',
                    enable=f'between(t,{start},{end})'
                )

            # 本番環境用のコード（コメントアウト） これは消さない！！
            '''
            for idx, (start, end) in enumerate(image_display_times):
                img_path = f"/tmp/temp_image_{idx}.png"
                await asyncio.to_thread(images[idx].save, img_path)
                img_input = ffmpeg.input(img_path)
                
                video = video.overlay(
                    img_input,
                    x='(main_w-overlay_w)/2',
                    y='(main_h-overlay_h)/2',
                    enable=f'between(t,{start},{end})'
                )
            '''
        
            font_path = os.path.join(current_dir, '..', 'font.ttf')
            font_path = os.path.abspath(font_path)

            for idx, (text, start, end) in enumerate(scene_timestamps):
                escaped_text = escape_ffmpeg_text(text)
                video = video.drawtext(
                    text=escaped_text,
                    fontfile=font_path,
                    fontsize=24,
                    fontcolor='white',
                    box=1,
                    boxcolor='black@0.8',
                    boxborderw=10,
                    x='(w-text_w)/2',
                    y='h-text_h-60',
                    enable=f'between(t,{start},{end})',
                    line_spacing=5
                )

            fade_duration = 2.0  # 1秒から2秒に変更

            video = video.filter('fade', type='in', duration=fade_duration)
            video = video.filter('fade', type='out', start_time=total_duration-fade_duration, duration=fade_duration)

            audio_input = ffmpeg.input(mixed_audio_path)

            audio_input = audio_input.filter('afade', type='in', duration=fade_duration)
            audio_input = audio_input.filter('afade', type='out', start_time=total_duration-fade_duration, duration=fade_duration)

            audio_input = (
                audio_input
                .filter('afftdn', nr=2, nt='w', om='o')  # ノイズ除去フィルター
                # .filter('highpass', f=200)  # 低周波ノイズを除去 (削除)
                # .filter('lowpass', f=3000)  # 高周波ノイズを除去 (削除)
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
