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
    non_silences = [{"Start": (start/1000),"Stop": (stop/1000), "Speech": True} for start,stop in non_silences]

    sentences = len(non_silences)
    
    assert sentences != 0, "No audible part detected."

    silences = silence.detect_silence(myaudio, min_silence_len=SILENCE_LEN, silence_thresh=dBFS-SILENCE_THRESH)
    silences = [{"Start": (start/1000),"Stop": (stop/1000), "Speech": False} for start,stop in silences]

    chunks = non_silences + silences
    chunks = sorted(chunks, key=lambda x: x['Start'])
    #print(chunks)
    segments = len(chunks)
 
    for i in range(segments):
        if chunks[i]["Speech"]:
            clip = myaudio[chunks[i]["Start"]*1000:chunks[i]["Stop"]*1000]
            chunks[i]["Path"] = f"AudioChunks/{i}.wav"
            clip.export(f"AudioChunks/{i}.wav", format="wav")

    return chunks, segments, sentences

def transcript(chunks, segments, SOURCE_LANG, sentences):

    whispermodel = whisper.load_model('small')
    text = []
    for i in range(segments):
        if chunks[i]["Speech"]:
            if SOURCE_LANG:
                transcription = whispermodel.transcribe(f"AudioChunks/{i}.wav", language = SOURCE_LANG)
            else:
                transcription = whispermodel.transcribe(f"AudioChunks/{i}.wav")

            if transcription['text'] == "":
                os.remove(f"AudioChunks/{i}.wav")
                chunks[i]["Speech"] = False 
                sentences -= 1
            else:
                chunks[i]["text"] = transcription['text'].strip()
                #print(f"Text: {transcription['text']}\n")
                text.append({"text": transcription['text'].strip(), "duration": chunks[i]['Stop'] - chunks[i]['Start']})
    #print(text)
    if sentences == 0:
        raise AssertionError("No speech detected.")
    gc.collect()
    torch.cuda.empty_cache()
    return chunks, text, sentences


def translation(text, chunks, segments, SOURCE_LANG, sentences):
    count = 0
    translated_text = []
    for i in range(segments):
        if chunks[i]["Speech"]:
            if SOURCE_LANG:
                completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": f"{text[count]}"}],
                    functions=[
                    {
                        "name": "translate",
                        "description": f"Translate the provided text from {SOURCE_LANG} to English while keeping the sentence the same length. Maintain equal speaking time. Consider the context of the previous conversation for your response.",
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
            else:
                completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": f"{text[count]}"}],
                    functions=[
                    {
                        "name": "translate",
                        "description": "Translate the provided text to English while keeping the sentence the same length. Maintain equal speaking time. Consider the context of the previous conversation for your response.",
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
            #print(data['translated_text'])
            translated_text.append(data['translated_text'])
            count += 1
            
    assert len(translated_text) == sentences, f"Sentences were {sentences}, GPT returned {len(translated_text)}"
    
    count = 0
    for i in range(segments):
        if chunks[i]["Speech"]:
            chunks[i]['translation'] = translated_text[count]
            print(f"Orignial: {chunks[i]['text']}\nTranslated: {chunks[i]['translation']}\n")
            count += 1

    return chunks

def audio_synthesis(chunks, segments, preset, CUSTOM_VOICE_NAME = "custom"):    
    tts = TextToSpeech(kv_cache=True, half=True)
    path = "AudioChunks"
    
    custom_voice_folder = f"tortoise/voices/{CUSTOM_VOICE_NAME}"
    if os.path.exists(custom_voice_folder):
        shutil.rmtree(custom_voice_folder)
    os.makedirs(custom_voice_folder)
    
    files = [f for f in os.listdir(path) if f.endswith('.wav')]
    files.sort(key=lambda f: os.path.getsize(os.path.join(path, f)), reverse=True)
    files = files[:3]
    for i, filename in enumerate(files):
        file_path = os.path.join(path, filename)
        with open(file_path, 'rb') as file:
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
        if chunks[i]["Speech"]:
            gen = tts.tts_with_preset(chunks[i]["translation"], voice_samples=voice_samples, conditioning_latents=conditioning_latents,preset=preset)
            torchaudio.save(f"ClonedAudio/{i}_generated.wav", gen.squeeze(0).cpu(), SAMPLE_RATE, bits_per_sample=16)

    print("Audio Generated")
    #print(segments)
    for i in range(segments):
        if chunks[i]["Speech"]:
            if i != segments - 1:
                #print(i)
                original_audio = AudioSegment.from_wav(f"AudioChunks/{i}.wav")
                #print(f"Original: {original_audio.duration_seconds}")
                next_silence_duration = chunks[i+1]["Stop"] - chunks[i+1]["Start"]
                #print(f"Silence: {next_silence_duration}")
                gen_audio = AudioSegment.from_wav(f"ClonedAudio/{i}_generated.wav")
                #print(f"Generated: {gen_audio.duration_seconds}")
                if gen_audio.duration_seconds < original_audio.duration_seconds:
                    addsilence = AudioSegment.silent(duration=(original_audio.duration_seconds - gen_audio.duration_seconds) * 1000)
                    gen_audio = gen_audio + addsilence

                elif gen_audio.duration_seconds < original_audio.duration_seconds + next_silence_duration:
                    new_silence_duration = (original_audio.duration_seconds + next_silence_duration) - gen_audio.duration_seconds
                    chunks[i+1]['Start'] = chunks[i+1]['Stop'] - new_silence_duration
                
                else:
                    with WavReader(f"ClonedAudio/{i}_generated.wav") as reader:
                        with WavWriter(f"ClonedAudio/{i}_generatedspeed.wav", reader.channels, reader.samplerate) as writer:
                            tsm = wsola(reader.channels, speed = (gen_audio.duration_seconds/(original_audio.duration_seconds + next_silence_duration)))
                            tsm.run(reader, writer)
                    gen_audio = AudioSegment.from_wav(f"ClonedAudio/{i}_generatedspeed.wav")
                    chunks[i+1]["Stop"] = chunks[i+1]["Start"]

                chunks[i]['Stop'] = chunks[i]['Start'] + gen_audio.duration_seconds


            else:
                original_audio = AudioSegment.from_wav(f"AudioChunks/{i}.wav")
                gen_audio = AudioSegment.from_wav(f"ClonedAudio/{i}_generated.wav")
                if gen_audio.duration_seconds < original_audio.duration_seconds:
                    addsilence = AudioSegment.silent(duration=(original_audio.duration_seconds - gen_audio.duration_seconds) * 1000)
                    gen_audio = gen_audio + addsilence
                else:
                    with WavReader(f"ClonedAudio/{i}_generated.wav") as reader:
                        with WavWriter(f"ClonedAudio/{i}_generatedspeed.wav", reader.channels, reader.samplerate) as writer:
                            tsm = wsola(reader.channels, speed= (gen_audio.duration_seconds/original_audio.duration_seconds))
                            tsm.run(reader, writer)
                    gen_audio = AudioSegment.from_wav(f"ClonedAudio/{i}_generatedspeed.wav")
        else:
            gen_audio = AudioSegment.silent(duration=(chunks[i]["Stop"] - chunks[i]["Start"])* 1000)
            ##print(f"Silence: {audio.duration_seconds}\tVideo: {chunks[i]['Stop'] - chunks[i]['Start']}\n")
            
        gen_audio.export(f"ModifiedAudio/{i}_adjusted_audio.wav", format="wav")


    segment = AudioSegment.from_wav("ModifiedAudio/0_adjusted_audio.wav")
    for i in range(1, segments):
        gen_audio = AudioSegment.from_wav(f"ModifiedAudio/{i}_adjusted_audio.wav")
        segment += gen_audio

    gc.collect()
    torch.cuda.empty_cache()

    background = AudioSegment.from_wav("Data/background.wav")
    overlayed_audio = background.overlay(segment)
    overlayed_audio.export(f"Full_Audio.wav", format="wav")

def lipsync(video_path, gan=True):
    video_file = os.path.join(os.getcwd(), video_path)
    audio_file = os.path.join(os.getcwd(),"Full_Audio.wav")
    result_file = os.path.join(os.getcwd(),"output.mp4")
    if gan:
        checkpoint = "checkpoints/wav2lip_gan.pth"
    else:
        checkpoint = "checkpoints/wav2lip.pth"
    os.system(f"cd Wav2Lip && python inference.py --checkpoint_path {checkpoint} --face {video_file} --audio {audio_file} --outfile {result_file}")
    gc.collect()
    torch.cuda.empty_cache()
    
def non_lipsync(video_path):
    audio_path = "Full_Audio.wav"
    output_path = "output.mp4"

    video_clip = VideoFileClip(video_path)
    audio_clip = AudioFileClip(audio_path)
    
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
                text = chunks[i]['translation']

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
        output_path
        ]
	subprocess.run(command, check=True)
	print("Subtitles merged successfully!")


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
