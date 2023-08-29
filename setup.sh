#!/bin/bash
mkdir Data
pip install audiotsm
pip install audio-separator
sudo apt-get install libsndfile1 -y
pip install --upgrade --no-deps --force-reinstall git+https://github.com/openai/whisper.git
pip install tiktoken
pip install moviepy
pip install pydub
sudo apt install gcc -y
pip install pytorch torchvision torchaudio pytorch-cuda==11.7 -c pytorch -c nvidia
git clone https://github.com/neonbjb/tortoise-tts.git
cd tortoise-tts
pip3 install tqdm
pip3 install rotary_embedding_torch
pip3 install transformers==4.31.0
pip3 install tokenizers
pip3 install inflect
pip3 install progressbar
pip3 install einops==0.4.1
pip3 install unidecode
pip3 install scipy
pip3 install librosa==0.9.1
pip3 install ffmpeg
pip3 install numpy
pip3 install numba
pip3 install torchaudio
pip3 install threadpoolctl
pip3 install llvmlite
pip3 install appdirs
pip3 install nbconvert==5.3.1
pip3 install tornado==4.2
pip3 install pydantic==1.9.1
pip3 install deepspeed
pip3 install py-cpuinfo
pip3 install hjson
pip3 install psutil
python3 setup.py install
pip install librosa
cd ..
mv tortoise-tts/tortoise tortoise
rm -rf tortoise-tts 
mkdir ClonedAudio
mkdir ModifiedAudio
mkdir AudioChunks

env OPENAI_API_KEY=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
pip install openai

git clone https://github.com/PrabalS12/DeepFake-LipSync-AI-using-Wav2Lip.git
mv DeepFake-LipSync-AI-using-Wav2Lip Wav2Lip
wget 'https://iiitaphyd-my.sharepoint.com/personal/radrabha_m_research_iiit_ac_in/_layouts/15/download.aspx?share=EdjI7bZlgApMqsVoEUUXpLsBxqXbn5z8VTmoxp55YNDcIA' -O 'Wav2Lip/checkpoints/wav2lip_gan.pth'
wget "https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth" -O "Wav2Lip/face_detection/detection/sfd/s3fd.pth"
pip install https://raw.githubusercontent.com/AwaleSajil/ghc/master/ghc-1.0-py3-none-any.whl
cd Wav2Lip && pip install -r requirements.txt
pip install -q youtube-dl
pip install ffmpeg-python
pip install librosa==0.9.1
pip install opencv-python
apt-get update && apt-get install libgl1 -y
cd ..

pip install pymongo
pip install boto3
