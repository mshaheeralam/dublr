import whisper
from moviepy.editor import VideoFileClip, AudioFileClip, CompositeVideoClip, clips_array, ColorClip
import moviepy.video.fx.all as vfx
import torchaudio
from pydub import AudioSegment, silence
import os
from tortoise.api import TextToSpeech
from tortoise.utils.audio import load_voice, get_voices, load_audio
import shutil
import openai
import gc
import torch
import glob
import subprocess
import json
from audiotsm import wsola
from audiotsm.io.wav import WavReader, WavWriter
from audio_separator import Separator

SILENCE_LEN= 500
SILENCE_THRESH = 20
SAMPLE_RATE = 22050 

openai.api_key = "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"

def padding(video_path, child):
    if not video_path.endswith('mp4'):
        cmd = ['ffmpeg', '-i', video_path, '-q:v', '0', 'Data/input_video.mp4']
        subprocess.run(cmd)
        #os.remove(video_path)
        video_path = 'Data/input_video.mp4'
    
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
    separator = Separator(audio_path, model_name='UVR-MDX-NET-Voc_FT', denoise_enabled=False, output_dir="Data", use_cuda=True)
    primary_stem_path, secondary_stem_path = separator.separate()
    os.rename(f"Data/{primary_stem_path}", "Data/vocals.wav")
    os.rename(f"Data/{secondary_stem_path}", "Data/background.wav")

def create_segments(audio_path):
    myaudio = AudioSegment.from_wav(audio_path)
    dBFS= myaudio.dBFS

    non_silences = silence.detect_nonsilent(myaudio, min_silence_len=SILENCE_LEN, silence_thresh=dBFS-SILENCE_THRESH)
    non_silences = [{"Start": (start/1000),"Stop": (stop/1000), "Speech": True, "Synthesis": True, "Duration": (stop/1000) - (start/1000)} for start,stop in non_silences]

    sentences = len(non_silences)
    
    assert sentences != 0, "No audible part detected."

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
    print("")
    for i in range(segments):
        print(chunks[i])

    return chunks, segments, sentences

def transcript(chunks, segments, SOURCE_LANG, sentences, audio_path):

    whispermodel = whisper.load_model('small')
    myaudio = AudioSegment.from_wav(audio_path)
    text = []
    count = 0
    while count < segments:
        if chunks[count]['Speech']:
            if SOURCE_LANG:
                transcription = whispermodel.transcribe(chunks[count]["Path"], language = SOURCE_LANG)
            else:
                transcription = whispermodel.transcribe(chunks[count]["Path"])

            # if transcription['text'] == "":
            #     os.remove(chunks[count]["Path"])
            #     chunks[count]["Speech"] = False
            #     print(chunks[count]) 
            #     sentences -= 1
            #     count += 1
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
                    # print(count)
                    # print(chunks[count])
                    # print(count-1)
                    # print(chunks[count-1])
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
                text.append(transcription['text'].strip())
                count += 1

    #print(text)
    text = "\n".join(text)

    if sentences == 0:
        raise AssertionError("No speech detected.")
    gc.collect()
    torch.cuda.empty_cache()


    print("")
    for i in range(segments):
        print(chunks[i])

    return chunks, text, segments, sentences


def translation(text, chunks, segments, SOURCE_LANG):
    for i in range(segments):
        #print(chunks[i])
        if chunks[i]['Speech']:
            if SOURCE_LANG:
                completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": f"Sentence: {chunks[i]['Text']}"}],
                    functions=[
                    {
                        "name": 'translate',
                        "description": f"""Translates a sentence from {SOURCE_LANG} to English ensuring:
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
            else:
                completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": f"Sentence: {chunks[i]['Text']}"}],
                    functions=[
                    {
                        "name": "translate",
                        "description": F""""Translates a sentence from to English ensuring:
                            - The translated length is similar to the original.
                            - Modern English vocabulary is used, avoiding outdated terms.
                            - The translation maintains the original context and meaning.""",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "translated_text": {
                                    "type": "string",
                                    "description": "Translated text string"
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
            print(f"Orignial: {chunks[i]['Text']}\nTranslated: {chunks[i]['Translation']}\nStart: {chunks[i]['Start']}\tStop: {chunks[i]['Stop']}\n")

    return chunks

def audio_synthesis(chunks, segments, preset, CUSTOM_VOICE_NAME = "custom"):    
    tts = TextToSpeech(kv_cache=True, half=True)
    
    custom_voice_folder = f"tortoise/voices/{CUSTOM_VOICE_NAME}"
    if os.path.exists(custom_voice_folder):
        shutil.rmtree(custom_voice_folder)
    os.makedirs(custom_voice_folder)
    
    file_path = sorted(chunks, key=lambda x: x['Duration'], reverse=True)
    files = [(chunk['Path']) for chunk in file_path if chunk['Synthesis'] and chunk['Speech']]
    files = files[:3]
    if not files:
        files = [(chunk['Path']) for chunk in file_path if chunk['Speech']]
        files = files[:3]
    #print(files)
    for i, filename in enumerate(files):
        with open(filename, 'rb') as file:
            file_data=file.read()
        new_file_path=os.path.join(custom_voice_folder,f'{i}.wav')
        with open(new_file_path,'wb') as f:
            f.write(file_data)
            
    voice_samples, _ = load_voice(CUSTOM_VOICE_NAME)
    voices = get_voices()
    cond_paths = voices[CUSTOM_VOICE_NAME]
    conds = []
    for cond_path in cond_paths:
        c = load_audio(cond_path, SAMPLE_RATE)
        conds.append(c)
    conditioning_latents = tts.get_conditioning_latents(conds)

    for i in range(segments):
        if chunks[i]['Speech']:
            gen = tts.tts_with_preset(chunks[i]["Translation"], voice_samples=None, conditioning_latents=conditioning_latents,preset=preset)
            torchaudio.save(f"ClonedAudio/{i}_generated.wav", gen.squeeze(0).cpu(), SAMPLE_RATE, bits_per_sample=16)

    print("Audio Generated\n")
    #print(segments)
    for i in range(segments):
        if chunks[i]['Speech']:
            original_audio_duration = chunks[i]['Stop'] - chunks[i]['Start']
            gen_audio = AudioSegment.from_wav(f"ClonedAudio/{i}_generated.wav")

            if gen_audio.duration_seconds < original_audio_duration:
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

    gc.collect()
    torch.cuda.empty_cache()

    background = AudioSegment.from_wav("Data/background.wav")
    overlayed_audio = background.overlay(segment)
    overlayed_audio.export(f"Full_Audio.wav", format="wav")

def lipsync(video_path, gan=True):
    video_file = os.path.join(os.getcwd(), video_path)
    audio_file = os.path.join(os.getcwd(),"Full_Audio.wav")
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
    output_path = "lipsync.mp4"

    video_clip = VideoFileClip(video_path)
    audio_clip = AudioFileClip(audio_path)
    
    if video_clip.rotation in (90, 270):
        video_clip = video_clip.resize(video_clip.size[::-1])
        video_clip.rotation = 0
	    
    # Mute the original video's audio
    video_clip = video_clip.set_audio(None)
    
    # Set your own audio
    video_with_new_audio = video_clip.set_audio(audio_clip)
    
    # Composite the video clip with the new audio
    final_clip = CompositeVideoClip([video_with_new_audio])
    
    # Write the output video with new audio
    final_clip.write_videofile(output_path, codec="libx264", fps=video_clip.fps)
    
    # Close the clips
    video_clip.close()
    audio_clip.close()
    final_clip.close()

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
                start_time = seconds_to_srt_time(chunks[i]['Start'])
                stop_time = seconds_to_srt_time(chunks[i]['Stop'])
                text = chunks[i]['Translation']

                srt_file.write(f"{count}\n")
                srt_file.write(f"{start_time} --> {stop_time}\n")
                srt_file.write(f"{text}\n")
                srt_file.write("\n")
                count+=1

def merge_srt_with_video(video_path, srt_filename, output_path):
	command = [
        'ffmpeg',
        '-i', video_path,
        '-vf', f"subtitles={srt_filename}",
        output_path,
        "-y"
        ]
	subprocess.run(command, check=True)
	print("Subtitles merged successfully!\n")


def remove_data(CUSTOM_VOICE_NAME = "custom"):
    folder_paths = [
        'Data/*',
        'ClonedAudio/*',
        'ModifiedAudio/*',
        'AudioChunks/*'
    ]
    if os.path.exists('Full_Audio.wav'):
        os.remove('Full_Audio.wav')

    if os.path.exists('subtitle.mp4'):
        os.remove('subtitle.mp4')

    for folder_path in folder_paths:
        files = glob.glob(folder_path)
        for f in files:
            os.remove(f)
    if os.path.exists(f"tortoise/voices/{CUSTOM_VOICE_NAME}"):
        shutil.rmtree(f"tortoise/voices/{CUSTOM_VOICE_NAME}")
