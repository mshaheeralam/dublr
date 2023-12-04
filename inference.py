import dublr as db
import time
import warnings
import os
warnings.filterwarnings('ignore')

def run(video_path, SOURCE_LANG=None, TAR_LANG=None, start=None, end=None):
    db.remove_data()
    try:
        start = time.time()
        video_path, audio_path = db.preprocess(video_path, start, end)
        end = time.time()
        if os.path.exists(audio_path):
            print("\nPreprocess :", round((end-start) / 60, 2), "min\n")
        else:
            raise FileNotFoundError("Could not preprocess")
         
        start = time.time()
        db.denoise(audio_path, start, end)
        end = time.time()
        if os.path.exists('Data/vocals.wav'):
            print("Denoise :", round((end-start) / 60, 2), "min\n")
        else:
            raise FileNotFoundError("Could not denoise")
        
        start = time.time()
        chunks = db.create_segments('Data/vocals.wav')
        end = time.time()
        if chunks:
            print("Chunks :", round((end-start) / 60, 2), "min\n")
        else:
            raise ValueError("Could not create segments")

        start = time.time()
        chunks = db.transcript(chunks, audio_path)
        end = time.time()
        if chunks:
            print("Transcription :", round((end-start) / 60, 2), "min\n")
        else:
            raise ValueError("Could not transcribe")

        start = time.time()
        chunks = db.translation(chunks, SOURCE_LANG, TAR_LANG)
        end = time.time()
        print("Translation :", round((end-start) / 60, 2), "min\n")
        
        start = time.time()
        chunks = db.audio_synthesis(chunks)
        end = time.time()
        if os.path.exists("ClonedAudio") and os.listdir("ClonedAudio"):
            print("Audio Generation :", round((end-start) / 60, 2), "min\n")
        else:
            raise FileNotFoundError("Could not synthesize audio")
        
        start = time.time()
        chunks = db.audio_modification(chunks)
        end = time.time()
        if os.path.exists("Full_Audio.wav") and os.path.exists("Full_vocals.wav"):
            print("Audio Modification :", round((end-start) / 60, 2), "min\n")
        else:
            raise FileNotFoundError("Could not modify audio")

        start = time.time()
        db.chunks_to_srt(chunks)
        end = time.time()
        if os.path.exists('Data/sub.srt'):
            print("Subtitles :", round((end-start) / 60, 2), "min\n")
        else:
            raise FileNotFoundError("Could not Generate SRT")

        start = time.time()
        video_path = db.non_lipsync(video_path)
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
    run(video_path='Videos/messi.mp4', SOURCE_LANG="Spanish", TAR_LANG="English")
    end = time.time()
    print("Total :", round((end-start) / 60, 2), "min\n")