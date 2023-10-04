from dublr import *
import time
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_audio, ffmpeg_extract_subclip
from moviepy.tools import subprocess_call
from moviepy.config import get_setting
import multiprocessing
import warnings
import os
warnings.filterwarnings('ignore')

def run(video_path, lip_sync, subtitiles, audio_syn=True, video=True, gan=True, SOURCE_LANG=None, start=None, end=None):
    remove_data()
    try:
        audio_path = video_path
        if video:
            if not video_path.endswith('mp4'):
                cmd = [get_setting("FFMPEG_BINARY"), "-y",
                       '-i', video_path, 
                       '-q:v', '0', 
                       'Data/input_video.mp4'
                       ]
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
            #pad_parent, pad_child = multiprocessing.Pipe()
            #pad = multiprocessing.Process(target=padding, args=(video_path, pad_child))
            #pad.start()

        start = time.time()
        denoise(audio_path)
        end = time.time()

        if os.path.exists('Data/vocals.mp3'):
            print("\nDenoise :", round((end-start) / 60, 2), "min\n")
        else:
            raise FileNotFoundError("Could not Denoise")
        
        # start = time.time()
        # multi_speaker = speaker_detection(audio_path)
        # end = time.time()

        # if not multi_speaker:
        #     print("Speaker Detection :", (end-start) / 60, "min\n")
        # else:
        #     raise AssertionError("Multiple speaker detected")

        start = time.time()
        chunks, segments, sentences = create_segments('Data/vocals.mp3')
        end = time.time()
        if chunks and segments and sentences:
            print("Chunks :", round((end-start) / 60, 2), "min\n")
        else:
            raise ValueError("Could Not Create Segments")

        start = time.time()
        chunks, segments, sentences = transcript(chunks, segments, SOURCE_LANG, sentences, audio_path)
        end = time.time()
        
        if sentences != 0:
            print("Transcription :", round((end-start) / 60, 2), "min\n")
        else:
            raise ValueError("Could Not Transcribe")

        start = time.time()
        chunks = translation(chunks, segments, SOURCE_LANG)
        end = time.time()
        print("Translation :", round((end-start) / 60, 2), "min\n")

        if audio_syn:
            start = time.time()
            audio_synthesis(chunks, segments)
            end = time.time()
            if os.path.exists("ClonedAudio") and os.listdir("ClonedAudio"):
                print("Audio Generation :", round((end-start) / 60, 2), "min\n")
            else:
                raise FileNotFoundError("Could not Synthesize Audio")
            
            start = time.time()
            audio_modification(chunks, segments)
            end = time.time()
            if os.path.exists("Full_Audio.wav") and os.path.exists("Full_vocals.wav"):
                print("Audio Modification :", round((end-start) / 60, 2), "min\n")
            else:
                raise FileNotFoundError("Could not modify Audio")

        if video and audio_syn:
            #video_path = pad_parent.recv()
            #pad.join()
            if lip_sync:
                start = time.time()
                video_path = lipsync(video_path, gan)
                end = time.time()
                print("Lip Sync :", round((end-start) / 60, 2), "min\n")
            
            start = time.time()
            video_path = non_lipsync(video_path)
            end = time.time()

            if os.path.exists(video_path):
                print("Merged :", round((end-start) / 60, 2), "min\n")
            else:
                raise FileNotFoundError("No Video File Found")
            

        if subtitiles:
            start = time.time()
            chunks_to_srt(chunks, segments, 'Data/sub.srt')
            if os.path.exists('Data/sub.srt'):
                print("SRT Generated\n")
            else:
                raise FileNotFoundError("Could not Generate SRT")
            
            merge_srt_with_video(video_path, 'Data/sub.srt', 'output.mp4')
            os.remove(video_path)
            video_path = 'output.mp4'
            if os.path.exists(video_path):
                print("Subtitled Video Generated\n")
            else:
                raise FileNotFoundError("No Video File Found")
            end = time.time()
            print("Subtitles :", (end-start) / 60, "min\n")
        else:
            os.rename("merged.mp4","output.mp4")
        # remove_data()
        return None
    except Exception as e:
        print(str(e))
        # remove_data()
        return e

if __name__ == '__main__':
    start = time.time()
    run(video_path='Videos/romi.mov', lip_sync=False, subtitiles=True, SOURCE_LANG="Urdu")
    end = time.time()
    print("Total :", round((end-start) / 60, 2), "min\n")
# ,start=12,end=56.260633
