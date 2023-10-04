#!/bin/bash

mkdir Data
pip install git+https://github.com/openai/whisper.git
pip install transformers==4.29.2
pip install --upgrade moviepy
pip3 install -U scipy

pip install pydub
sudo apt install gcc -y
git clone https://github.com/152334H/tortoise-tts-fast
mv tortoise-tts-fast tortoise-tts
cd tortoise-tts
pip install -r requirements.txt
pip install -e .
pip install git+https://github.com/152334H/BigVGAN.git
cd ..
mv tortoise-tts/tortoise tortoise
mv tortoise-tts/static static
rm -rf tortoise-tts 
mkdir ClonedAudio
mkdir ModifiedAudio
mkdir AudioChunks

env OPENAI_API_KEY=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
env ORGANIZATION_ID=org-1ODchwRlVvrDaOmHpC07IcVa
pip install openai
pip install requests

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

pip install git+https://github.com/Desklop/RNNoise_Wrapper
sudo apt-get install autoconf libtool -y
git clone https://github.com/Desklop/RNNoise_Wrapper
cd RNNoise_Wrapper
chmod +x compile_rnnoise.sh
apt install unzip
apt install make
./compile_rnnoise.sh
cd ..
