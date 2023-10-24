from dublr import *
import time
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_audio, ffmpeg_extract_subclip
from moviepy.tools import subprocess_call
from moviepy.config import get_setting
import warnings
import os
warnings.filterwarnings('ignore')

def run(video_path, SOURCE_LANG=None, TAR_LANG=None, start=None, end=None):
    remove_data()
    try:
        audio_path = video_path
        if not video_path.endswith('mp4'):
            cmd = [get_setting("FFMPEG_BINARY"), "-y", '-i', video_path, '-q:v', '0', 'Data/input_video.mp4']
            subprocess_call(cmd)
            #os.remove(video_path)
            video_path = 'Data/input_video.mp4'
        if start is not None and end is not None:
            ffmpeg_extract_subclip(video_path, start, end, "Data/cropped_input_video.mp4")
            video_path = "Data/cropped_input_video.mp4"
        if os.path.exists(video_path):
            print("Video File Exists\n")
        else:
            raise FileNotFoundError("No Video File Found")

        ffmpeg_extract_audio(video_path, "Data/input_video.wav", bitrate=SAMPLE_RATE)
        audio_path = "Data/input_video.wav"
        
        if os.path.exists(audio_path):
            print("Audio File Exists\n")
        else:
            raise FileNotFoundError("No Audio File Found")
 
        start = time.time()
        denoise(audio_path)
        end = time.time()
        if os.path.exists('Data/vocals.wav'):
            print("\nDenoise :", round((end-start) / 60, 2), "min\n")
        else:
            raise FileNotFoundError("Could not Denoise")
        
        start = time.time()
        chunks, segments, sentences = create_segments('Data/vocals.wav')
        end = time.time()
        if chunks and segments and sentences:
            print("Chunks :", round((end-start) / 60, 2), "min\n")
        else:
            raise ValueError("Could not create segments")

        start = time.time()
        chunks, segments = transcript(chunks, segments, audio_path)
        end = time.time()
        if sentences != 0:
            print("Transcription :", round((end-start) / 60, 2), "min\n")
        else:
            raise ValueError("Could not transcribe")

        start = time.time()
        chunks = translation(chunks, segments, SOURCE_LANG, TAR_LANG)
        end = time.time()
        print("Translation :", round((end-start) / 60, 2), "min\n")

        input("\n\nCHECK TRANSCRIPTIONS. PRESS ENTER TO CONTINUE\n\n\n")
        
        start = time.time()
        audio_synthesis(chunks, segments)
        end = time.time()
        if os.path.exists("ClonedAudio") and os.listdir("ClonedAudio"):
            print("Audio Generation :", round((end-start) / 60, 2), "min\n")
        else:
            raise FileNotFoundError("Could not synthesize audio")
        
        start = time.time()
        audio_modification(chunks, segments)
        end = time.time()
        if os.path.exists("Full_Audio.wav") and os.path.exists("Full_vocals.wav"):
            print("Audio Modification :", round((end-start) / 60, 2), "min\n")
        else:
            raise FileNotFoundError("Could not modify audio")

        start = time.time()
        chunks_to_srt(chunks, segments, 'Data/sub.srt')
        end = time.time()
        if os.path.exists('Data/sub.srt'):
            print("Subtitles :", (end-start) / 60, "min\n")
        else:
            raise FileNotFoundError("Could not Generate SRT")
 
        start = time.time()
        video_path = non_lipsync(video_path)
        end = time.time()

        if os.path.exists(video_path):
            print("Merged :", round((end-start) / 60, 2), "min\n")
        else:
            raise FileNotFoundError("No Video File Found")

        return None
    except Exception as e:
        print(str(e))
        return e

if __name__ == '__main__':
    start = time.time()
    run(video_path='Videos/lesson8.1.mp4', SOURCE_LANG="English", TAR_LANG="Urdu")
    end = time.time()
    print("Total :", round((end-start) / 60, 2), "min\n")
# ,start=12,end=56.260633
