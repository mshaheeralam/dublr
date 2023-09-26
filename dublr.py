import whisper
from moviepy.editor import VideoFileClip, clips_array, ColorClip
from moviepy.tools import subprocess_call
from moviepy.config import get_setting
import moviepy.video.fx.all as vfx
from pydub import AudioSegment, silence
import os
from TTS.api import TTS
import openai
import gc
import torch
import glob
import json
from audiotsm import wsola
from audiotsm.io.wav import WavReader, WavWriter
# from audio_separator import Separator
from pyannote.audio import Pipeline
from dotenv import load_dotenv
import nltk
nltk.download('punkt') 
from nltk.tokenize import sent_tokenize
from aeneas.tools.execute_task import ExecuteTaskCLI

load_dotenv()

SILENCE_LEN= 500
SILENCE_THRESH = 20
SAMPLE_RATE = 22050 
INAUDIBLE = "[INAUDIBLE]"
openai.api_key = os.getenv("OPENAIKEY")

LANG = {
    'ar': 'Arabic',
    'eu': 'Basque',
    'be': 'Belarusian',
    'bn': 'Bengali',
    'ca': 'Catalan',
    'zh': 'Chinese',
    'hr': 'Croatian',
    'cs': 'Czech',
    'da': 'Danish',
    'nl': 'Dutch',
    'en': 'English',
    'fi': 'Finnish',
    'fr': 'French',
    'de': 'German',
    'el': 'Greek',
    'he': 'Hebrew',
    'hi': 'Hindi',
    'hu': 'Hungarian',
    'ga': 'Irish',
    'it': 'Italian',
    'ja': 'Japanese',
    'ko': 'Korean',
    'la': 'Latin',
    'fa': 'Persian',
    'pl': 'Polish',
    'pt': 'Portuguese',
    'ru': 'Russian',
    'sr': 'Serbian',
    'es': 'Spanish',
    'tr': 'Turkish',
    'ts': 'Tsonga',
    'ur': 'Urdu'}

def padding(video_path, child):
    clip = VideoFileClip(video_path, audio=False)
    
    if clip.rotation in (90, 270):
        clip = clip.resize(clip.size[::-1])
        clip.rotation = 0
    
    if clip.aspect_ratio < 1:
        print("Vertical video found. Changing to 16:9\n")
        pad = ColorClip(size=clip.size, color=(0, 0, 0), duration=clip.duration)
        clips = [[pad, clip, pad]]
        stacked = clips_array(clips)
        clip = vfx.resize(stacked, (1280, 720))
        clip.write_videofile("Data/padded.mp4", audio=False)
        video_path = "Data/padded.mp4"
    
    child.send(video_path)
    child.close()

def denoise(audio_path):
    cmd = ["python3", "-m", "demucs.separate", "-o", 'Data/', '--mp3', '--mp3-preset', '2', f'--mp3-bitrate={SAMPLE_RATE}', '--two-stems=vocals', audio_path]
    subprocess_call(cmd)
    os.rename("Data/htdemucs/input_video/vocals.mp3", "Data/vocals.mp3")
    os.rename("Data/htdemucs/input_video/no_vocals.mp3", "Data/background.mp3")
    os.system("rm -rf Data/htdemucs")
    os.system("ffmpeg -i Data/vocals.mp3 Data/vocals.wav -y")
    os.system("ffmpeg -i Data/background.mp3 Data/background.wav -y")

def speaker_detection(audio_path):
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization@2.1",
                                    use_auth_token="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")
    diarization = pipeline(audio_path)
    #print(diarization)
    for _, _, speaker in diarization.itertracks(yield_label=True):
        if int(speaker[-2:]) > 0:
            return True

    return False
        
def create_segments(audio_path):
    myaudio = AudioSegment.from_mp3(audio_path)
    dBFS= myaudio.dBFS

    non_silences = silence.detect_nonsilent(myaudio, min_silence_len=SILENCE_LEN, silence_thresh=dBFS-SILENCE_THRESH)
    non_silences = [{"Start": (start/1000),"Stop": (stop/1000), "Speech": True, "Synthesis": True, "Speech Duration": (stop/1000)-(start/1000)} for start,stop in non_silences]

    sentences = len(non_silences)
    
    if sentences == 0:
        return None, None, None

    chunks = sorted(non_silences, key=lambda x: x['Start'])
    segments = len(chunks)
    # for i in range(segments):
    #     print(chunks[i])
    
    count = 0
    for i in range(segments):
        if chunks[count]['Stop'] - chunks[count]['Start'] < 0.5:
            if count == 0 and segments > 1:
                chunks[count+1]['Start'] = chunks[count]['Start']    
            else:
                chunks[count-1]['Stop'] = chunks[count]['Stop']

            del chunks[count]
            segments -= 1
        else:
            count += 1
    # print("")
    # for i in range(segments):
    #     print(chunks[i])

    for i in range(segments):  
        clip = myaudio[chunks[i]["Start"]*1000:chunks[i]["Stop"]*1000]
        chunks[i]["Path"] = f"AudioChunks/{i}.wav"
        clip.export(f"AudioChunks/{i}.wav", format="wav")

    for i in range(segments-1):
        chunks[i]['Stop'] = chunks[i+1]['Start']

    chunks[-1]['Stop'] = myaudio.duration_seconds
    # print("")
    # for i in range(segments):
    #     print(chunks[i])

    return chunks, segments, sentences

def transcript(chunks, segments, SOURCE_LANG, sentences, audio_path):

    whispermodel = whisper.load_model('small')
    myaudio = AudioSegment.from_wav(audio_path)
    count = 0
    while count < segments:
        if chunks[count]['Speech']:
            audio = whisper.load_audio(chunks[count]["Path"])
            audio = whisper.pad_or_trim(audio)
            mel = whisper.log_mel_spectrogram(audio).to(whispermodel.device)
            _, probs = whispermodel.detect_language(mel)
            chunks[count]['Language'] = max(probs, key=probs.get)

            transcription = whispermodel.transcribe(chunks[count]["Path"], language = LANG.get(chunks[count]['Language'], SOURCE_LANG))

            if len(transcription["text"]) < 10:
                if count == 0 and segments > 1:
                    chunks[count+1]['Start'] = chunks[count]['Start']
                    os.remove(chunks[count]["Path"])
                    os.remove(chunks[count+1]["Path"])
                    clip = myaudio[chunks[count+1]["Start"]*1000:chunks[count+1]["Stop"]*1000]
                    clip.export(chunks[count+1]["Path"], format="wav")
                    del chunks[count]
                    chunks[count]['Synthesis'] = False
                else:
                    chunks[count]['Start'] = chunks[count-1]['Start']
                    os.remove(chunks[count]["Path"])
                    os.remove(chunks[count-1]["Path"])
                    clip = myaudio[chunks[count]["Start"]*1000:chunks[count]["Stop"]*1000]
                    clip.export(chunks[count]["Path"], format="wav")
                    del chunks[count-1]
                    count -= 1
                    chunks[count]['Synthesis'] = False
                segments -= 1
            else:
                chunks[count]["Text"] = transcription['text'].strip()
                #print(f"Text: {transcription['text']}\n")
                count += 1
    count = 0
    while count < segments:
        if chunks[count]['Speech']:
            chunks[count]['Duration'] = chunks[count]['Stop'] - chunks[count]['Start']
            if len(chunks[count]["Text"]) / (chunks[count]['Stop'] - chunks[count]['Start']) > 25:
                chunks[count]["Text"] = INAUDIBLE
        count += 1

    gc.collect()
    torch.cuda.empty_cache()


    # print("")
    # for i in range(segments):
    #     print(chunks[i])

    return chunks, segments, sentences

def translation(chunks, segments, SOURCE_LANG):
    for i in range(segments):
        #print(chunks[i])
        if chunks[i]['Speech']:
            if chunks[i]['Language'] != 'en' and chunks[i]['Text'] != INAUDIBLE:
                if SOURCE_LANG or LANG.get(chunks[i]['Language']):
                    prompt = f" from {LANG.get(chunks[i]['Language'], SOURCE_LANG)}"
                else:
                    prompt = ""

                completion = openai.ChatCompletion.create(
                    model="gpt-4",
                    messages=[{"role": "user", "content": f"Sentence: {chunks[i]['Text']}"}],
                        functions=[
                        {
                            "name": 'translate',
                            "description": f"""Translates a sentence{prompt} to English ensuring:
                                - The translated length is similar to the original.
                                - Modern English vocabulary is used, avoiding outdated terms.
                                - The translation maintains the original context and meaning.""",
                            "parameters": {
                                "type": 'object',
                                "properties": {
                                    "translated_text": {
                                        'type': 'string',
                                        'description': 'Translated text string'
                                    }
                                },
                                "required": ["translated_text"]
                            }
                        }
                        ],
                        function_call={"name": "translate"},
                    )
                reply_content = completion.choices[0].message
                data = reply_content.to_dict()['function_call']['arguments']
                data = json.loads(data)
                chunks[i]['Translation'] = data['translated_text']
            else:
                print(f"Not Translated: {chunks[i]['Text']}")
                chunks[i]['Translation'] = chunks[i]['Text']
            print(f"Orignial: {chunks[i]['Text']}\nTranslated: {chunks[i]['Translation']}\nLanguage: {chunks[i]['Language']}\nStart: {chunks[i]['Start']}\tStop: {chunks[i]['Stop']}\n")
    return chunks

def audio_synthesis(chunks, segments):    
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v1", gpu=True)

    vocals = AudioSegment.from_wav("Data/vocals.wav")

    file_path = sorted(chunks, key=lambda x: x['Speech Duration'], reverse=True)
    files = [chunk['Path'] for chunk in file_path if chunk['Synthesis'] and chunk['Speech'] and chunk['Speech Duration'] > 5]
   
    if not files:
        files = [chunk['Path'] for chunk in file_path if chunk['Speech'] and chunk['Speech Duration'] > 5]
        if not files:
            files = [chunk['Path'] for chunk in file_path if chunk['Speech'] and chunk['Duration'] > 5]
            if not files:
                clip = vocals[file_path[0]["Start"]*1000:file_path[0]["Stop"]*1000]
                for i in range(1,len(file_path[:1])):
                    clip += vocals[file_path[i]["Start"]*1000:file_path[i]["Stop"]*1000]
                clip.export("Data/Speech.wav", format="wav")
                files = ['Data/Speech.wav']

    files = files[0]
    print("\nFILE: ", files)
    
    for i in range(segments):
        print(f"Speech Duration: {round(file_path[i]['Speech Duration'], 2)}\tChunk Duration: {round(file_path[i]['Duration'], 2)}\tFile: {file_path[i]['Path']}")
    
    # print(FILE_PATH)
    # for i in range(segments):
    #     print(file_path)

    # audio = AudioSegment.from_wav(files)
    # cond_len = 0
    # if int(audio.duration_seconds) // 3 > 3:
    #     cond_len = int(audio.duration_seconds) // 3
    # else:
    #     cond_len = 3
    # print("\ncond len: ",cond_len)
    
    for i in range(segments):
        if chunks[i]['Speech']:
            if chunks[i]['Text'] != INAUDIBLE:
                try:
                    tts.tts_to_file(text=chunks[i]["Translation"], file_path=f"ClonedAudio/{i}_generated.wav", 
                                    speaker_wav=files, language="en", gpt_cond_len=3, decoder_iterations=150,
                                    temperature=1e-05, top_p=0.95, top_k=100)
                except:
                    print("CUDA OFFFFFFFFFFFFFF\n")
                    exit()
    gc.collect()
    torch.cuda.empty_cache()

def audio_modification(chunks, segments):

    for i in range(segments):
        if chunks[i]['Speech'] and chunks[i]['Text'] != INAUDIBLE:
            original_audio_duration = chunks[i]['Stop'] - chunks[i]['Start']
            gen_audio = AudioSegment.from_wav(f"ClonedAudio/{i}_generated.wav")

            if gen_audio.duration_seconds < original_audio_duration:
                chunks[i]['Stop'] = chunks[i]['Start'] + gen_audio.duration_seconds
                addsilence = AudioSegment.silent(duration=(original_audio_duration - gen_audio.duration_seconds) * 1000)
                gen_audio = gen_audio + addsilence
            else:
                with WavReader(f"ClonedAudio/{i}_generated.wav") as reader:
                    with WavWriter(f"ClonedAudio/{i}_generatedspeed.wav", reader.channels, reader.samplerate) as writer:
                        tsm = wsola(reader.channels, speed = (gen_audio.duration_seconds/original_audio_duration))
                        tsm.run(reader, writer)
                gen_audio = AudioSegment.from_wav(f"ClonedAudio/{i}_generatedspeed.wav")
        else:
            gen_audio = AudioSegment.silent(duration=(chunks[i]["Stop"] - chunks[i]["Start"])* 1000)
            
        gen_audio.export(f"ModifiedAudio/{i}_adjusted_audio.wav", format="wav")


    segment = AudioSegment.silent(duration=(chunks[0]['Start'] - 0.0) * 1000)
    for i in range(segments):
        gen_audio = AudioSegment.from_wav(f"ModifiedAudio/{i}_adjusted_audio.wav")
        segment += gen_audio

    # full_audio = AudioSegment.from_wav('Data/vocals.wav')
    # segment += AudioSegment.silent(duration=(full_audio.duration_seconds - chunks[-1]['Stop']) * 1000)
    background = AudioSegment.from_wav("Data/background.wav")
    if segment.duration_seconds < background.duration_seconds:
        addsilence = AudioSegment.silent(duration=(background.duration_seconds - segment.duration_seconds) * 1000) 
        segment = segment + addsilence
    
    segment.export("Full_vocals.wav", format="wav")   
    overlayed_audio = background.overlay(segment)
    overlayed_audio.export(f"Full_Audio.wav", format="wav")
    print(f"Vocals: {segment.duration_seconds}\nOverlayed: {overlayed_audio.duration_seconds}\nOriginal: {background.duration_seconds}\n")

    gc.collect()
    torch.cuda.empty_cache()

def lipsync(video_path, gan=True):
    video_file = os.path.join(os.getcwd(), video_path)
    audio_file = os.path.join(os.getcwd(),"Full_vocals.wav")
    result_file = os.path.join(os.getcwd(),"lipsync.mp4")
    if gan:
        checkpoint = "checkpoints/wav2lip_gan.pth"
    else:
        checkpoint = "checkpoints/wav2lip.pth"
    os.system(f"cd Wav2Lip && python inference.py --checkpoint_path {checkpoint} --face {video_file} --audio {audio_file} --outfile {result_file}")
    gc.collect()
    torch.cuda.empty_cache()
    
    return "lipsync.mp4"

def non_lipsync(video_path):
    audio_path = "Full_Audio.wav"
    output_path = "merged.mp4"
    cmd = [
        get_setting("FFMPEG_BINARY"), "-y",
        "-i", video_path,
        "-i", audio_path,
        "-map", "0:v",
        "-map", "1:a",
        output_path
    ]

    subprocess_call(cmd)
  
    return output_path

def seconds_to_srt_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    milliseconds = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"

def chunks_to_srt(chunks, segments, srt_filename):
    with open(srt_filename, 'w') as srt_file:
        count = 1	
        for i in range(segments):
            if chunks[i]["Speech"]:
                sentences = sent_tokenize(chunks[i]["Translation"])
                with open("Data/sentence.txt", "w", encoding='utf-8') as file:
                    file.write('\n'.join(sentences))

                data_path = "Data/manifest.json"
                ExecuteTaskCLI(use_sys=False).run(arguments=[
                    None,
                    f"ModifiedAudio/{i}_adjusted_audio.wav",
                    "Data/sentence.txt",
                    u"task_language=eng|is_text_type=plain|os_task_file_format=json",
                    data_path
                ])
    
                with open(data_path, "r") as f:
                    data = json.load(f)
                # json_trans = json.dumps(data, indent = 1, ensure_ascii=False)
                # f = open(data_path, "w", encoding='utf-8')
                # f.write(json_trans)
                # f.close()
                # with open(data_path, "r", encoding="utf-8") as f:
                #     data = json.load(f)
                #print(data['fragments'])
                for j in range(len(data["fragments"])):
                    
                    start_time = seconds_to_srt_time(chunks[i]['Start'] + float(data["fragments"][j]["begin"]))
                    if j == len(data['fragments']) - 1:
                        stop_time = seconds_to_srt_time(chunks[i]['Stop'])
                    else:
                        stop_time = seconds_to_srt_time(chunks[i]['Start'] + float(data["fragments"][j]["end"]))
                    text = data["fragments"][j]["lines"][0]

                    srt_file.write(f"{count}\n")
                    srt_file.write(f"{start_time} --> {stop_time}\n")
                    srt_file.write(f"{text}\n")
                    srt_file.write("\n")
                    count+=1
    gc.collect()
    torch.cuda.empty_cache()
    
def merge_srt_with_video(video_path, srt_filename, output_path):
    cmd = [
        get_setting("FFMPEG_BINARY"), "-y",
        '-i', video_path,
        '-vf', f"subtitles={srt_filename}:force_style='Fontname=Consolas,BackColour=&H80000000,Spacing=0.2,Outline=0,Shadow=0.75'",
        output_path,
        "-y"
        ]
    subprocess_call(cmd)
    print("Subtitles merged successfully!\n")

def remove_data():
    folder_paths = [
        'Data/*',
        'ClonedAudio/*',
        'ModifiedAudio/*',
        'AudioChunks/*'
    ]
    if os.path.exists('Full_Audio.wav'):
        os.remove('Full_Audio.wav')

    if os.path.exists('Full_vocals.wav'):
        os.remove('Full_vocals.wav')

    if os.path.exists('subtitle.mp4'):
        os.remove('subtitle.mp4')

    if os.path.exists('lipsync.mp4'):
        os.remove('lipsync.mp4')

    for folder_path in folder_paths:
        files = glob.glob(folder_path)
        for f in files:
            os.remove(f)

