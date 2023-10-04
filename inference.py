from dublr import *
import time
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_audio
from moviepy.video.io.VideoFileClip import VideoFileClip
import multiprocessing
import os

def run(video_path, lip_sync, preset, gan, SOURCE_LANG=None, start=None, end=None):
    try:
        if start is not None and end is not None:
            codec = "libx264"
            video = VideoFileClip(video_path).subclip(start, end)
            file_extension = video_path.split(".")[-1]
            video.write_videofile("Data/cropped_input_video."+file_extension,codec=codec)
            video_path="Data/cropped_input_video."+file_extension
        if os.path.exists(video_path):
            print("Video File Exists")
        else:
            raise FileNotFoundError("No Video File Found")

        ffmpeg_extract_audio(video_path, "Data/input_video.wav", bitrate=SAMPLE_RATE)
        if os.path.exists("Data/input_video.wav"):
            print("Audio Extracted")
        else:
            raise FileNotFoundError("No Audio File Found")
        
        #pad_parent, pad_child = multiprocessing.Pipe()
        #pad = multiprocessing.Process(target=padding, args=(video_path, pad_child))
        #pad.start()

        start = time.time()
        denoise()
        end = time.time()
        
        if os.path.exists('Data/test_denoised.wav'):
            print("Denoise :", (end-start) / 60, "min")
        else:
            raise FileNotFoundError("Could not Denoise")

        start = time.time()
        chunks, segments, sentences = create_segments('Data/test_denoised.wav')
        end = time.time()
        if chunks and segments and sentences:
            print("Chunks :", (end-start) / 60, "min")
        else:
            raise ValueError("Could Not Create Segments")

        start = time.time()
        chunks, text, sentences = transcript(chunks, segments, SOURCE_LANG, sentences)
        end = time.time()
        if text:
            print("Transcription :", (end-start) / 60, "min")
        else:
            raise ValueError("Could Not Transcribe")

        start = time.time()
        chunks = translation(text, chunks, segments, SOURCE_LANG, sentences)
        end = time.time()
        print("Translation :", (end-start) / 60, "min")

        start = time.time()
        audio_synthesis(chunks, segments, preset)
        end = time.time()
        if os.path.exists("ModifiedAudio/Full_Audio.wav"):
            print("Audio :", (end-start) / 60, "min")
        else:
            raise FileNotFoundError("Could not Synthesize Audio")

        #video_path = pad_parent.recv()
        #pad.join()

        if lip_sync:
            start = time.time()
            lipsync(video_path, gan)
            end = time.time()
            print("Lip Sync :", (end-start) / 60, "min")
        else:
            start = time.time()
            non_lipsync(video_path)
            end = time.time()
            print("Non Lip Sync :", (end-start) / 60, "min")

        remove_data()
        return None
    except Exception as e:
        print(str(e))
        #remove_data()
        return e

if __name__ == '__main__':
	run('input_video.mp4', lip_sync=True, preset='ultra_fast', gan=True, SOURCE_LANG='urdu',start=0,end=5)	

