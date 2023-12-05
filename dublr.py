from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_audio, ffmpeg_extract_subclip
from moviepy.tools import subprocess_call
from moviepy.config import get_setting
from pydub import AudioSegment, silence
import os
from openai import OpenAI
import gc
import torch
import torchaudio
import glob
import time
import json
from audiotsm import wsola
from audiotsm.io.wav import WavReader, WavWriter
from dotenv import load_dotenv
import nltk
nltk.download('punkt') 
from nltk.tokenize import sent_tokenize
from aeneas.tools.execute_task import ExecuteTaskCLI
from elevenlabs import set_api_key, clone, generate, play, voices
import requests
import boto3
import uuid

load_dotenv()

SILENCE_LEN = 500
SILENCE_THRESH = 20
SAMPLE_RATE = 22050 
INAUDIBLE = "[INAUDIBLE]"

client = OpenAI(api_key=os.getenv("OPENAIKEY"))

set_api_key(os.getenv("11LABs"))

def preprocess(video_path, start, end):
    if not video_path.endswith('mp4'):
        cmd = [get_setting("FFMPEG_BINARY"), "-y", '-i', video_path, '-q:v', '0', 'Data/input_video.mp4']
        subprocess_call(cmd)
        #os.remove(video_path)
        video_path = 'Data/input_video.mp4'

    if start is not None and end is not None:
        ffmpeg_extract_subclip(video_path, start, end, "Data/cropped_input_video.mp4")
        video_path = "Data/cropped_input_video.mp4"

    ffmpeg_extract_audio(video_path, "Data/input_video.wav", bitrate=SAMPLE_RATE)
    audio_path = "Data/input_video.wav"
    
    return video_path, audio_path

def denoise(audio_path):
    cmd = ["python", "-m", "demucs.separate", "-o", 'Data/', '--mp3', '--mp3-preset', '2', f'--mp3-bitrate={SAMPLE_RATE}', '--two-stems=vocals', audio_path]
    subprocess_call(cmd)
    os.rename("Data/htdemucs/input_video/vocals.mp3", "Data/vocals.mp3")
    os.rename("Data/htdemucs/input_video/no_vocals.mp3", "Data/background.mp3")
    os.system("rm -rf Data/htdemucs")
    cmd = [get_setting("FFMPEG_BINARY"), "-y", '-i', 'Data/vocals.mp3', 'Data/vocals.wav']
    subprocess_call(cmd)
    cmd = [get_setting("FFMPEG_BINARY"), "-y", '-i', 'Data/background.mp3', 'Data/background.wav']
    subprocess_call(cmd)

    return 'Data/vocals.wav'
        
def create_segments(audio_path):
    myaudio = AudioSegment.from_wav(audio_path)
    dBFS= myaudio.dBFS

    non_silences = silence.detect_nonsilent(myaudio, min_silence_len=SILENCE_LEN, silence_thresh=dBFS-SILENCE_THRESH)
    non_silences = [{"Start": (start/1000),"Stop": (stop/1000), "Speech": True, "Synthesis": True, "Duration": (stop/1000) - (start/1000)} for start,stop in non_silences]

    sentences = len(non_silences)
    
    if sentences == 0:
        return None

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

    return chunks

def transcript(chunks, audio_path):

    myaudio = AudioSegment.from_wav(audio_path)
    count = 0
    segments = len(chunks)

    while count < segments:
        if chunks[count]['Speech']:
        
            transcription = client.audio.transcriptions.create(model="whisper-1", file=open(chunks[count]["Path"], "rb"), response_format="text")
            print(transcription)
            if len(transcription) < 10:
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
            else:
                chunks[count]["Text"] = transcription.strip()
                #print(f"Text: {transcription['text']}\n")
                count += 1
    count = 0
    while count < segments:
        if chunks[count]['Speech']:
            chunks[count]['Duration'] = chunks[count]['Stop'] - chunks[count]['Start']
            if len(chunks[count]["Text"]) // (chunks[count]['Stop'] - chunks[count]['Start']) > 25:
                chunks[count]["Text"] = INAUDIBLE
        count += 1

    gc.collect()
    torch.cuda.empty_cache()


    # print("")
    # for i in range(segments):
    #     print(chunks[i])

    return chunks


def translation(chunks, SOURCE_LANG, TAR_LANG):

    segments = len(chunks)

    for i in range(segments):
        if chunks[i]['Speech']:
            if chunks[i]['Text'] != INAUDIBLE:
                if SOURCE_LANG:
                    description = f"from {SOURCE_LANG} to {TAR_LANG}"
                else:
                    description = f"to {TAR_LANG} "
                completion = client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": f"Sentence: {chunks[i]['Text']}"}],
                    functions=[
                    {
                        "name": 'translate',
                        "description": f"""Act as an expert translator and translate the text {description} ensuring:
                            - The translated length is similar to the original.
                            - Modern {TAR_LANG} vocabulary is used, avoiding outdated terms.
                            - Use simple English words, instead of translating to difficult {TAR_LANG} words. 
                            - The translation maintains the original context and meaning.
                            - Do not translate filler words.""",
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

                print(completion.choices[0].message)
                reply_content = completion.choices[0].message
                data = reply_content.function_call.arguments
                data = json.loads(data)
                print(data)
                chunks[i]['Translation'] = data['translated_text']
            else:
                chunks[i]['Translation'] = chunks[i]['Text']
            
            #print(f"Original: {chunks[i]['Text']}\nTranslated: {chunks[i]['Translation']}\nPath: {chunks[i]['Path']}\nStart: {chunks[i]['Start']}\tStop: {chunks[i]['Stop']}\n\n")
    

    return chunks

def audio_synthesis(chunks, name="test", stability = 0.5, similarity_boost = 0.75, style = 0.0, boost = True, cloning = True, voice = None):    
    
    files = [os.path.join("AudioChunks", file) for file in os.listdir("AudioChunks")]

    if cloning:
        voice = clone(name=name, files=files)

    url = f'https://api.elevenlabs.io/v1/text-to-speech/{voice.voice_id}'
    headers = {
        'accept': 'audio/mpeg',
        'xi-api-key': os.getenv("11LABs"),
        'Content-Type': 'application/json',
    }

    segments = len(chunks)

    for i in range(segments):
        if chunks[i]['Speech']:
            if chunks[i]['Translation'] != INAUDIBLE:
                data = {
                    "text": chunks[i]["Translation"],
                    "voice_settings": {
                        "stability": stability,
                        "similarity_boost": similarity_boost,
                        "style": style, "use_speaker_boost": boost
                    },
                    "model_id": "eleven_multilingual_v2",
                }
                response = requests.post(url, headers=headers, json=data)

                if response.status_code == 200:
                    if chunks[i].get("Generated Path"):
                        output_file_path = chunks[i]["Generated Path"]
                    else:
                        chunks[i]["Generated Path"] = f"ClonedAudio/{i}_generated.wav"
                        output_file_path = chunks[i]["Generated Path"]
                    with open(output_file_path[:-3] + "mp3", "wb") as f:
                        f.write(response.content)
                    print(f"Audio file saved as '{output_file_path}'")
                    cmd = [get_setting("FFMPEG_BINARY"), "-y", '-i', output_file_path[:-3] + "mp3", chunks[i]["Generated Path"]]
                    subprocess_call(cmd)
                else:
                    print(f"Error: {response.status_code} - {response.text}")

    return chunks, voice

def audio_modification(chunks):

    segments = len(chunks)

    for i in range(segments):
        if chunks[i]['Speech'] and chunks[i]['Translation'] != INAUDIBLE:
            original_audio_duration = chunks[i]['Stop'] - chunks[i]['Start']
            gen_audio = AudioSegment.from_wav(chunks[i]["Generated Path"])

            if gen_audio.duration_seconds < original_audio_duration:
                chunks[i]['Stop'] = chunks[i]['Start'] + gen_audio.duration_seconds
                addsilence = AudioSegment.silent(duration=(original_audio_duration - gen_audio.duration_seconds) * 1000)
                gen_audio = gen_audio + addsilence
            else:
                with WavReader(chunks[i]["Generated Path"]) as reader:
                    with WavWriter(f"ClonedAudio/{i}_generatedspeed.wav", reader.channels, reader.samplerate) as writer:
                        tsm = wsola(reader.channels, speed = (gen_audio.duration_seconds/original_audio_duration))
                        tsm.run(reader, writer)
                gen_audio = AudioSegment.from_wav(f"ClonedAudio/{i}_generatedspeed.wav")
        else:
            gen_audio = AudioSegment.silent(duration=(chunks[i]["Stop"] - chunks[i]["Start"])* 1000)
        
        if chunks[i].get("Modified Path"):
            gen_audio.export(chunks[i]["Modified Path"], format="wav")
        else:
            chunks[i]["Modified Path"] = f"ModifiedAudio/{i}_adjusted_audio.wav"
            gen_audio.export(chunks[i]["Modified Path"], format="wav")
            

    return chunks

def lipsync(video_path, audio_path):
    s3 = boto3.client('s3')
    unique_video = str(uuid.uuid4().hex) + '-' + video_path
    s3.upload_file(video_path, "dublr-lip-sync-bucket", unique_video)
    video_url = s3.generate_presigned_url('get_object', Params={'Bucket': "dublr-lip-sync-bucket", 'Key': unique_video}, ExpiresIn=3600)

    unique_audio = str(uuid.uuid4().hex) + '-' + audio_path
    s3.upload_file(audio_path, "dublr-lip-sync-bucket", unique_audio)
    audio_url = s3.generate_presigned_url('get_object', Params={'Bucket': "dublr-lip-sync-bucket", 'Key': unique_audio}, ExpiresIn=3600)
    
    url = "https://api.synclabs.so/video"
    headers = {
        'accept': 'application/json',
        'x-api-key': os.getenv('SYNC'),
        'Content-Type': 'application/json'
    }

    data = {
        "audioUrl": audio_url,
        "videoUrl": video_url,
        "synergize": True
    }

    response = requests.post(url, headers=headers, json=data)

    if response.json().get('id'):
        output = response.json()['id']
    else:
        return
    
    url = f"https://api.synclabs.so/video/{output}"
    while True:
        response = requests.get(url, headers=headers)
        print("response", response) 
        if response.json().get('url'):
            break
        else:
            print(response.json())
            time.sleep(20)

    print(response.json())

    s3.delete_object(Bucket="dublr-lip-sync-bucket", Key=unique_video)
    s3.delete_object(Bucket="dublr-lip-sync-bucket", Key=unique_audio)

    if response.json().get('url'):
        os.system(f"wget -O lipsync.mp4 {response.json()['url']}")
    else:
        return
    
    gc.collect()
    torch.cuda.empty_cache()
    
    return "lipsync.mp4"

def seconds_to_srt_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    milliseconds = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"

def chunks_to_srt(chunks, srt_filename='Data/sub.srt'):

    segments = len(chunks)

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

def non_lipsync(video_path, chunks):
    segment = AudioSegment.silent(duration=(chunks[0]['Start'] - 0.0) * 1000)
    segments = len(chunks)
    for i in range(segments):
        gen_audio = AudioSegment.from_wav(f"ModifiedAudio/{i}_adjusted_audio.wav")
        segment += gen_audio

    # full_audio = AudioSegment.from_wav('Data/vocals.wav')
    # segment += AudioSegment.silent(duration=(full_audio.duration_seconds - chunks[-1]['Stop']) * 1000)

    segment.export("Full_vocals.wav", format="wav")   
    background = AudioSegment.from_wav("Data/background.wav")
    overlayed_audio = background.overlay(segment)
    overlayed_audio.export(f"Full_Audio.wav", format="wav")
    
    gc.collect()
    torch.cuda.empty_cache()
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

    for folder_path in folder_paths:
        files = glob.glob(folder_path)
        for f in files:
            os.remove(f)
