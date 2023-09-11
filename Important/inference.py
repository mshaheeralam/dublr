from dublr import *
import time
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_audio
from moviepy.video.io.VideoFileClip import VideoFileClip
import multiprocessing

def run(video_path, lip_sync, audio_syn, subtitiles, preset, gan, video, SOURCE_LANG=None, start=None, end=None):
    try:
        audio_path = video_path
        if video:
            if start is not None and end is not None:
                video = VideoFileClip(video_path).subclip(start, end)
                file_extension = video_path.split(".")[-1]
                video.write_videofile("Data/cropped_input_video." + file_extension, codec="libx264", fps=video.fps)
                video_path = "Data/cropped_input_video." + file_extension
            
            if os.path.exists(video_path):
                print("Video File Exists")
            else:
                raise FileNotFoundError("No Video File Found")

            
            ffmpeg_extract_audio(video_path, "Data/input_video.wav", bitrate=SAMPLE_RATE)
            audio_path = "Data/input_video.wav"
            
            if os.path.exists(audio_path):
                print("Audio File Exists")
            else:
                raise FileNotFoundError("No Audio File Found")
            #pad_parent, pad_child = multiprocessing.Pipe()
            #pad = multiprocessing.Process(target=padding, args=(video_path, pad_child))
            #pad.start()

        start = time.time()
        denoise(audio_path)
        end = time.time()
        if os.path.exists('Data/vocals.wav'):
            print("Denoise :", (end-start) / 60, "min")
        else:
            raise FileNotFoundError("Could not Denoise")

        start = time.time()
        chunks, segments, sentences = create_segments('Data/vocals.wav')
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
        

        if audio_syn:
            start = time.time()
            audio_synthesis(chunks, segments, preset)
            end = time.time()
            if os.path.exists("Full_Audio.wav"):
                print("Audio :", (end-start) / 60, "min")
            else:
                raise FileNotFoundError("Could not Synthesize Audio")

        if subtitiles:
            start = time.time()
            chunks_to_srt(chunks, segments, 'Data/sub.srt')
            if os.path.exists('Data/sub.srt'):
                print("SRT Generated")
            else:
                raise FileNotFoundError("Could not Generate SRT")
            if audio_syn:
                if os.path.exists(video_path):
                    print("Video File Exists")
                else:
                    raise FileNotFoundError("No Video File Found")
                merge_srt_with_video(video_path, 'Data/sub.srt', 'Data/subtitle.mp4')
                video_path = 'Data/subtitle.mp4'
                if os.path.exists('Data/subtitle.mp4'):
                    print("Subtitled Video Generated")
                else:
                    raise FileNotFoundError("No Video File Found")
            else:
                if os.path.exists(video_path):
                    print("Video File Exists")
                else:
                    raise FileNotFoundError("No Video File Found")
                merge_srt_with_video(video_path, 'Data/sub.srt', 'output.mp4')
                video_path = 'output.mp4'
                if os.path.exists(video_path):
                    print("Subtitled Video Generated")
                else:
                    raise FileNotFoundError("No Video File Found")
            end = time.time()
            print("Subtitles :", (end-start) / 60, "min")

        if video:
            #video_path = pad_parent.recv()
            #pad.join()
            if audio_syn:
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
        remove_data()
        return e

if __name__ == '__main__':
    run('Ronaldo.mp4', False, True, True, 'fast', False, True, 'Russian', 0, 30)
