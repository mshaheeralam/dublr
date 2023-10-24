from moviepy.tools import subprocess_call
from moviepy.config import get_setting
from pydub import AudioSegment, silence
import os
import openai
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

openai.api_key = os.getenv("OPENAIKEY")
set_api_key(os.getenv("11LABs"))

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

        
def create_segments(audio_path):
    myaudio = AudioSegment.from_wav(audio_path)
    dBFS= myaudio.dBFS

    non_silences = silence.detect_nonsilent(myaudio, min_silence_len=SILENCE_LEN, silence_thresh=dBFS-SILENCE_THRESH)
    non_silences = [{"Start": (start/1000),"Stop": (stop/1000), "Speech": True, "Synthesis": True, "Duration": (stop/1000) - (start/1000)} for start,stop in non_silences]

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

def transcript(chunks, segments, audio_path):

    myaudio = AudioSegment.from_wav(audio_path)
    count = 0
    while count < segments:
        if chunks[count]['Speech']:
        
            transcription = openai.Audio.transcribe("whisper-1", open(chunks[count]["Path"], "rb"))

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
            if len(chunks[count]["Text"]) // (chunks[count]['Stop'] - chunks[count]['Start']) > 25:
                chunks[count]["Text"] = INAUDIBLE
        count += 1

    gc.collect()
    torch.cuda.empty_cache()


    # print("")
    # for i in range(segments):
    #     print(chunks[i])

    return chunks, segments


def translation(chunks, segments, SOURCE_LANG, TAR_LANG):

    manifest = {"data": []}

    for i in range(segments):
        if chunks[i]['Speech']:
            if chunks[i]['Text'] != INAUDIBLE:
                if SOURCE_LANG:
                    description = f"Translates a sentence from {SOURCE_LANG} to {TAR_LANG} ensuring:"
                else:
                    description = f"Translates a sentence from to {TAR_LANG} ensuring:"
                completion = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[{"role": "user", "content": f"Sentence: {chunks[i]['Text']}"}],
                    functions=[
                    {
                        "name": 'translate',
                        "description": f"""{description}
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
                chunks[i]['Translation'] = chunks[i]['Text']
            
            print(f"Original: {chunks[i]['Text']}\nTranslated: {chunks[i]['Translation']}\nPath: {chunks[i]['Path']}\nStart: {chunks[i]['Start']}\tStop: {chunks[i]['Stop']}\n\n")
            manifest["data"].append({"Original": chunks[i]['Text'], "Translated": chunks[i]['Translation'], "Path": chunks[i]['Path'], "Start": chunks[i]['Start'], "Stop": chunks[i]['Stop']})
    
    with open("Data/data.json", "w", encoding ='utf8') as file:
        json.dump(manifest, file, ensure_ascii = False, indent = 4)

    return chunks

def audio_synthesis(chunks, segments, email="test"):    

    background = AudioSegment.from_wav("Data/background.wav")

    if background.duration_seconds < 10:
        files = ['Data/vocals.wav']
    else:
        file_path = sorted(chunks, key=lambda x: x['Duration'], reverse=True)
        files = [(chunk['Path']) for chunk in file_path if chunk['Synthesis'] and chunk['Speech']]
        
        if not files:
            files = [(chunk['Path']) for chunk in file_path if chunk['Speech']]

    files = files[0]
    
    print(f"\nSample: {files}\n")
    
    voice = clone(name=email, files=[files])
    url = f'https://api.elevenlabs.io/v1/text-to-speech/{voice.voice_id}'
    headers = {
        'accept': 'audio/mpeg',
        'xi-api-key': os.getenv("11LABs"),
        'Content-Type': 'application/json',
    }

    with open("Data/data.json", "r") as file:
        data = json.load(file)

    for i in range(segments):
        if chunks[i]['Speech']:
            chunks[i]["Text"] = data["data"][i]["Original"]
            chunks[i]["Translation"] = data["data"][i]["Translated"]
            chunks[i]["Start"] = data["data"][i]["Start"]
            chunks[i]["Stop"] = data["data"][i]["Stop"]

    for i in range(segments):
        if chunks[i]['Speech']:
            if chunks[i]['Translation'] != INAUDIBLE:
                data = {
                    "text": chunks[i]["Translation"],
                    "voice_settings": {
                        "stability": 0.5,
                        "similarity_boost": 0.75,
                    },
                    "model_id": "eleven_monolingual_v1",
                }
                response = requests.post(url, headers=headers, json=data)

                if response.status_code == 200:
                    output_file_path = f"ClonedAudio/{i}_generated.mp3"
                    with open(output_file_path, "wb") as f:
                        f.write(response.content)
                    print(f"Audio file saved as '{output_file_path}'")
                    cmd = [get_setting("FFMPEG_BINARY"), "-y", '-i', f"ClonedAudio/{i}_generated.mp3", f"ClonedAudio/{i}_generated.wav"]
                    subprocess_call(cmd)
                else:
                    print(f"Error: {response.status_code} - {response.text}")

    gc.collect()
    torch.cuda.empty_cache()

def audio_modification(chunks, segments):
    for i in range(segments):
        if chunks[i]['Speech'] and chunks[i]['Translation'] != INAUDIBLE:
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

    segment.export("Full_vocals.wav", format="wav")   
    background = AudioSegment.from_wav("Data/background.wav")
    overlayed_audio = background.overlay(segment)
    overlayed_audio.export(f"Full_Audio.wav", format="wav")
    gc.collect()
    torch.cuda.empty_cache()

def lipsync(video_path, audio_path, gan=True):
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
