from dublr import *
import time
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_audio
from moviepy.video.io.VideoFileClip import VideoFileClip
import multiprocessing

def run(video_path, lip_sync, audio_syn, subtitiles, preset, gan, video, SOURCE_LANG=None, start=None, end=None):
    try:
        audio_path = video_path
        if video:
            if not video_path.endswith('mp4'):
                cmd = ['ffmpeg', '-i', video_path, '-q:v', '0', 'Data/input_video.mp4']
                subprocess.run(cmd)
                #os.remove(video_path)
                video_path = 'Data/input_video.mp4'
            
            clip = VideoFileClip(video_path, audio=False)
            
            if clip.rotation in (90, 270):
                clip = clip.resize(clip.size[::-1])
                clip.rotation = 0
            if start is not None and end is not None:
                video = VideoFileClip(video_path).subclip(start, end)
                file_extension = video_path.split(".")[-1]
                video.write_videofile("Data/cropped_input_video." + file_extension, codec="libx264", fps=video.fps)
                video_path = "Data/cropped_input_video." + file_extension
            
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
        if os.path.exists('Data/vocals.wav'):
            print("Denoise :", (end-start) / 60, "min\n")
        else:
            raise FileNotFoundError("Could not Denoise")

        start = time.time()
        chunks, segments, sentences = create_segments('Data/vocals.wav')
        end = time.time()
        if chunks and segments and sentences:
            print("Chunks :", (end-start) / 60, "min\n")
        else:
            raise ValueError("Could Not Create Segments")

        start = time.time()
        chunks, text, segments, sentences = transcript(chunks, segments, SOURCE_LANG, sentences, audio_path)
        end = time.time()
        if text:
            print("Transcription :", (end-start) / 60, "min\n")
        else:
            raise ValueError("Could Not Transcribe")

        start = time.time()
        chunks = translation(text, chunks, segments, SOURCE_LANG)
        end = time.time()
        print("Translation :", (end-start) / 60, "min\n")

        end = time.time()
        if audio_syn:
            start = time.time()
            audio_synthesis(chunks, segments, preset)
            end = time.time()
            if os.path.exists("Full_Audio.wav"):
                print("Audio :", (end-start) / 60, "min\n")
            else:
                raise FileNotFoundError("Could not Synthesize Audio")

        if video and audio_syn:
            #video_path = pad_parent.recv()
            #pad.join()
            if lip_sync:
                start = time.time()
                video_path = lipsync(video_path, gan)
                end = time.time()
                print("Lip Sync :", (end-start) / 60, "min\n")
            else:
                start = time.time()
                video_path = non_lipsync(video_path)
                end = time.time()
                print("Non Lip Sync :", (end-start) / 60, "min\n")

        if subtitiles:
            start = time.time()
            chunks_to_srt(chunks, segments, 'Data/sub.srt')
            if os.path.exists('Data/sub.srt'):
                print("SRT Generated\n")
            else:
                raise FileNotFoundError("Could not Generate SRT")
            
            if os.path.exists(video_path):
                print("Video File Exists\n")
            else:
                raise FileNotFoundError("No Video File Found")
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
            os.rename("lipsync.mp4","output.mp4")
        remove_data()
        return None
    except Exception as e:
        print(str(e))
        remove_data()
        return e

if __name__ == '__main__':
    run('Videos/Ronaldo2.mp4', False, True, True, 'fast', True, True, "Portuguese", 14.0, 30.0)
