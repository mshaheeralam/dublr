from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_audio, ffmpeg_extract_subclip
from moviepy.editor import VideoFileClip, CompositeVideoClip, ImageClip
from moviepy.tools import subprocess_call
from moviepy.config import get_setting
from pydub import AudioSegment, silence
import os
from openai import OpenAI
import gc
import torch
import glob
import json
from audiotsm import wsola
from audiotsm.io.wav import WavReader, WavWriter
from elevenlabs import set_api_key, clone
import requests
import spacy
import demucs.separate

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
    demucs.separate.main(["--mp3", "-o", "Data/", "--two-stems=vocals", f"--mp3-bitrate={22050}", audio_path])
    # cmd = ["python", "-m", "demucs.separate", "-o", 'Data/', '--mp3', '--mp3-preset', '2', f'--mp3-bitrate={SAMPLE_RATE}', '--two-stems=vocals', audio_path]
    # subprocess_call(cmd)
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
        chunks[i]["index"]=i
        clip.export(f"AudioChunks/{i}.wav", format="wav")

    for i in range(segments-1):
        chunks[i]['Stop'] = chunks[i+1]['Start']

    chunks[-1]['Stop'] = myaudio.duration_seconds
    # print("")
    # for i in range(segments):
    #     print(chunks[i])

    return chunks

def transcript(chunk):

    #client = OpenAI(api_key=key)

    if chunk['Speech']:
        transcription = client.audio.transcriptions.create(model="whisper-1", file=open(chunk["Path"], "rb"), response_format="text")
        chunk["Text"] = transcription.strip()
        #print(f"Text: {transcription['text']}\n")

    # print("")
    # for i in range(segments):
    #     print(chunks[i])

    return chunk

def modify_transcript(chunks, audio_path):

    segments = len(chunks)
    myaudio = AudioSegment.from_wav(audio_path)

    count = 0
    while len(chunks) > 1 and count < segments:
        if chunks[count]['Speech']:
            if len(chunks[count]["Text"]) < 10:
                if count == 0 and segments > 1:
                    chunks[count+1]['Start'] = chunks[count]['Start']
                    chunks[count+1]['Text'] = f"{chunks[count]['Text']} {chunks[count+1]['Text']}"
                    os.remove(chunks[count]["Path"])
                    os.remove(chunks[count+1]["Path"])
                    clip = myaudio[chunks[count+1]["Start"]*1000:chunks[count+1]["Stop"]*1000]
                    clip.export(chunks[count+1]["Path"], format="wav")
                    del chunks[count]
                    chunks[count]['Synthesis'] = False
                else:
                    chunks[count]['Start'] = chunks[count-1]['Start']
                    chunks[count]['Text'] = f"{chunks[count-1]['Text']} {chunks[count]['Text']}"
                    os.remove(chunks[count]["Path"])
                    os.remove(chunks[count-1]["Path"])
                    clip = myaudio[chunks[count]["Start"]*1000:chunks[count]["Stop"]*1000]
                    clip.export(chunks[count]["Path"], format="wav")
                    del chunks[count-1]
                    count -= 1
                    chunks[count]['Synthesis'] = False
                segments -= 1
            else:
                #print(f"Text: {transcription['text']}\n")
                count += 1

    count = 0
    while count < segments:
        if chunks[count]['Speech']:
            chunks[count]['Duration'] = chunks[count]['Stop'] - chunks[count]['Start']
            CPS = len(chunks[count]["Text"]) / (chunks[count]['Stop'] - chunks[count]['Start'])
            if CPS > 25.0:
                chunks[count]["Text"] = INAUDIBLE
            else:
                chunks[count]["CPS"] = CPS
        count += 1
    
    return chunks

def translation(chunk, SOURCE_LANG, TAR_LANG):

    #client = OpenAI(api_key=key)

    if SOURCE_LANG == 'English':
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(chunk['Text'])
        keywords = [token.text for token in doc if token.pos_ in ["NOUN", "PROPN"]]
        chunk["Keywords"] = ", ".join(keywords)
    
    if chunk['Speech']:
        if chunk['Text'] != INAUDIBLE:
            if SOURCE_LANG:
                description = f"from {SOURCE_LANG} to {TAR_LANG}"
            else:
                description = f"to {TAR_LANG} "
            completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": f"Sentence: {chunk['Text']}"}],
                functions=[
                {
                    "name": 'translate',
                    "description": f"""Act as an expert translator and translate the text {description} ensuring:
                        - The translated length is similar to the original.
                        - Modern {TAR_LANG} vocabulary is used, avoiding outdated terms.
                        - Use simple English words, instead of translating to difficult {TAR_LANG} words. 
                        - The translation maintains the original context and meaning.
                        - Do not translate filler words.
                        - Keep names in original text.
                        - The translated text must be under {chunk["Duration"]} for a speaker with {chunk["CPS"]} characters per second.""",
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
            data = reply_content.function_call.arguments
            data = json.loads(data)
            chunk['Translation'] = data['translated_text']
        else:
            chunk['Translation'] = chunk['Text']
            
            #print(f"Original: {chunks[i]['Text']}\nTranslated: {chunks[i]['Translation']}\nPath: {chunks[i]['Path']}\nStart: {chunks[i]['Start']}\tStop: {chunks[i]['Stop']}\n\n")
    
    return chunk

def audio_synthesis(chunks, name="test", stability = 0.5, similarity_boost = 0.75, style = 0.0, boost = True, cloning = True, voice = None):    
    
    #set_api_key(key)

    files = [os.path.join("AudioChunks", file) for file in os.listdir("AudioChunks")][:25]

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
    
    video = VideoFileClip(output_path)

    if abs(video.rotation) in (90, 270):
        video = video.resize(video.size[::-1])
        video.rotation = 0

    title = ImageClip("localised.png").set_duration(video.duration).set_pos(("center","center")).resize(height=min(video.size))

    final = CompositeVideoClip([video, title])
    final.write_videofile("output.mp4")

    return 'output.mp4'

def remove_data():
    os.system("rm -rf Data/htdemucs")

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
