#!/bin/bash
mkdir Data

pip install audiotsm
pip install -U demucs

pip install tiktoken
pip install moviepy
pip install pydub

mkdir ClonedAudio
mkdir ModifiedAudio
mkdir AudioChunks

pip install nltk
pip install elevenlabs
pip install openai

pip install pymongo
pip install boto3
pip install python-dotenv

sudo apt-get install -y espeak swig libespeak-dev
wget https://raw.githubusercontent.com/readbeyond/aeneas/master/install_dependencies.sh
bash install_dependencies.sh
pip install aeneas