import whisper
from moviepy.editor import VideoFileClip, AudioFileClip, CompositeVideoClip, clips_array, ColorClip
import moviepy.video.fx.all as vfx
import torchaudio
from pydub import AudioSegment, silence
import os
from tortoise.api import TextToSpeech
from tortoise.utils.audio import load_voice
import shutil
import openai
import gc
import torch
import glob
from rnnoise_wrapper import RNNoise
import subprocess
import json

SILENCE_LEN= 250
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

def denoise():
    denoiser = RNNoise()
    audio = denoiser.read_wav("Data/input_video.wav")
    denoised_audio = denoiser.filter(audio)
    denoiser.write_wav('Data/test_denoised.wav', denoised_audio, SAMPLE_RATE)

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
            clip.export(f"AudioChunks/{i}.wav", format="wav")

    return chunks, segments, sentences

def transcript(chunks, segments, SOURCE_LANG, sentences):
    gc.collect()
    torch.cuda.empty_cache()

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
    assert sentences != 0, "No speech detected."
    
    return chunks, text, sentences


def translation(text, chunks, segments, SOURCE_LANG, sentences):

    completion = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": f"{text}"}],
        functions=[
        {
            "name": "translate",
            "description": f"Translate the list of given text from {SOURCE_LANG} to English while preserving the duration in seconds of the corresponding sentence.",
            "parameters": {
                "type": "object",
                "properties": {
                    "translated_text": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "description": "Translated text string"
                        },
                        "description": "List of transltated text"
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
    text = data['translated_text']

    assert len(text) == sentences, f"Sentences were {sentences}, GPT returned {len(text)}"
    count = 0
    for i in range(segments):
        if chunks[i]["Speech"]:
            chunks[i]['translation'] = text[count]
            print(f"Orignial: {chunks[i]['text']}\nTranslated: {chunks[i]['translation']}\n")
            count += 1

    return chunks

def audio_synthesis(chunks, segments, preset = "fast", CUSTOM_VOICE_NAME = "custom"):
    gc.collect()
    torch.cuda.empty_cache()
    
    tts = TextToSpeech()
    path = "AudioChunks"
    custom_voice_folder = f"tortoise/voices/{CUSTOM_VOICE_NAME}"
    if os.path.exists(custom_voice_folder):
        shutil.rmtree(custom_voice_folder)
    os.makedirs(custom_voice_folder)

    files = [f for f in os.listdir(path) if f.endswith('.wav')][:3]
    for i, filename in enumerate(files):
        with open(os.path.join(path, filename), 'rb') as file:
            with open(os.path.join(custom_voice_folder, f'{i}.wav'), 'wb') as f:
                f.write(file.read())

    voice_samples, conditioning_latents = load_voice(CUSTOM_VOICE_NAME)
    for i in range(segments):
        if chunks[i]["Speech"]:
            gen = tts.tts_with_preset(chunks[i]["translation"], voice_samples=voice_samples, conditioning_latents=conditioning_latents,preset=preset)
            torchaudio.save(f"ClonedAudio/{i}_generated.wav", gen.squeeze(0).cpu(), SAMPLE_RATE)
            
    for i in range(segments):
        if chunks[i]["Speech"]:
            video = AudioSegment.from_wav(f"AudioChunks/{i}.wav")
            audio = AudioSegment.from_wav(f"ClonedAudio/{i}_generated.wav")
            if audio.duration_seconds > video.duration_seconds:
                audio = audio.speedup(playback_speed=float(audio.duration_seconds / video.duration_seconds))
            else:
                addsilence = AudioSegment.silent(duration=(video.duration_seconds - audio.duration_seconds) * 1000)
                audio = audio + addsilence
        else:
            audio = AudioSegment.silent(duration=(chunks[i]["Stop"] - chunks[i]["Start"])* 1000)
            ##print(f"Silence: {audio.duration_seconds}\tVideo: {chunks[i]['Stop'] - chunks[i]['Start']}\n")
            
        audio.export(f"ModifiedAudio/{i}_adjusted_audio.wav", format="wav")


    segment = AudioSegment.from_wav("ModifiedAudio/0_adjusted_audio.wav")
    for i in range(1, segments):
        audio = AudioSegment.from_wav(f"ModifiedAudio/{i}_adjusted_audio.wav")
        segment += audio
        
    segment.export(f"ModifiedAudio/Full_Audio.wav", format="wav")

def lipsync(video_path, gan=True):
    gc.collect()
    torch.cuda.empty_cache()

    video_file = os.path.join(os.getcwd(), video_path)
    audio_file = os.path.join(os.getcwd(),"ModifiedAudio/Full_Audio.wav")
    result_file = os.path.join(os.getcwd(),"output.mp4")
    if gan:
        checkpoint = "checkpoints/wav2lip_gan.pth"
    else:
        checkpoint = "checkpoints/wav2lip.pth"
    os.system(f"cd Wav2Lip && python inference.py --checkpoint_path {checkpoint} --face {video_file} --audio {audio_file} --outfile {result_file}")

def non_lipsync(video_path):
    
    audio_path = "ModifiedAudio/Full_Audio.wav"
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
    final_clip.write_videofile(output_path, codec="libx264")
    
    # Close the clips
    video_clip.close()
    audio_clip.close()
    final_clip.close()

    
def remove_data(CUSTOM_VOICE_NAME = "custom"):
    folder_paths = [
        'Data/*',
        'ClonedAudio/*',
        'ModifiedAudio/*',
        'AudioChunks/*'
    ]

    for folder_path in folder_paths:
        files = glob.glob(folder_path)
        for f in files:
            os.remove(f)
    shutil.rmtree(f"tortoise/voices/{CUSTOM_VOICE_NAME}")
