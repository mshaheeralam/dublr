#!/bin/bash
mkdir Data

pip install TTS
pip install audiotsm
pip install -U demucs

pip install --upgrade --no-deps --force-reinstall git+https://github.com/openai/whisper.git
pip install tiktoken
pip install moviepy
pip install pydub

mkdir ClonedAudio
mkdir ModifiedAudio
mkdir AudioChunks

pip install openai

git clone https://github.com/PrabalS12/DeepFake-LipSync-AI-using-Wav2Lip.git
mv DeepFake-LipSync-AI-using-Wav2Lip Wav2Lip
wget 'https://iiitaphyd-my.sharepoint.com/personal/radrabha_m_research_iiit_ac_in/_layouts/15/download.aspx?share=EdjI7bZlgApMqsVoEUUXpLsBxqXbn5z8VTmoxp55YNDcIA' -O 'Wav2Lip/checkpoints/wav2lip_gan.pth'
wget "https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth" -O "Wav2Lip/face_detection/detection/sfd/s3fd.pth"
pip install https://raw.githubusercontent.com/AwaleSajil/ghc/master/ghc-1.0-py3-none-any.whl
cd Wav2Lip && pip install -r requirements.txt
pip install -q youtube-dl
pip install ffmpeg-python
pip install opencv-python
apt-get update && apt-get install libgl1 -y

pip install pymongo
pip install boto3
pip install python-dotenv

pip install -qq https://github.com/pyannote/pyannote-audio/archive/refs/heads/develop.zip

sudo apt-get install -y espeak swig libespeak-dev
wget https://raw.githubusercontent.com/readbeyond/aeneas/master/install_dependencies.sh
bash install_dependencies.sh
pip install aeneas