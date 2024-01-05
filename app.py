import streamlit as st
import dublr as db
import time
import warnings
import os
import requests
import concurrent.futures
import traceback
import base64
import glob
from pydub import AudioSegment

warnings.filterwarnings('ignore')

LANG = ['Arabic', 'Chinese', 'English', 'French', 'German', 'Hindi', 'Italian', 'Russian', 'Spanish', 'Turkish', 'Urdu']

# for k, v in st.session_state.items():
#         st.session_state[k] = v

def remove_keys():
    for key in st.session_state.keys():
        #if key != 'openai_key' and key != 'elevenlabs_key':
        del st.session_state[key]

def remove():

    files = glob.glob("*.mp4")
    for f in files:
        os.remove(f)
        
    url = f"https://api.elevenlabs.io/v1/voices/{st.session_state.voice.voice_id}"
    headers = {'xi-api-key': os.getenv("11LABs", "b9fe99a3506bd81ba51162a99346db55"),}
    response = requests.request("DELETE", url, headers=headers)

    if response.status_code != 200:
        raise LookupError(f"Error: {response.status_code} - {response.text}")

    remove_keys()

def run():
    db.remove_data()
    try:
        if st.session_state.start_time == st.session_state.end_time:
            st.session_state.start_time = None
            st.session_state.end_time = None
            
        with st.spinner("Preprocessing..."):
            start = time.time()
            st.session_state.video_path, st.session_state.audio_path = db.preprocess(st.session_state.video_path, st.session_state.start_time, st.session_state.end_time)
            end = time.time()
            if os.path.exists(st.session_state.audio_path):
                print("\nPreprocess :", round((end-start) / 60, 2), "min\n")
            else:
                raise FileNotFoundError("Could not preprocess")

        with st.spinner("Denoising..."): 
            start = time.time()
            st.session_state.audio_path = db.denoise(st.session_state.audio_path)
            end = time.time()
            if os.path.exists(st.session_state.audio_path):
                print("Denoise :", round((end-start) / 60, 2), "min\n")
            else:
                raise FileNotFoundError("Could not denoise")
        
        with st.spinner("Segmenting..."):
            start = time.time()
            st.session_state.chunks = db.create_segments(st.session_state.audio_path)
            end = time.time()
            if st.session_state.chunks:
                print("Chunks :", round((end-start) / 60, 2), "min\n")
            else:
                raise ValueError("Could not create segments")

        with st.spinner("Transcribing..."):
            start = time.time()
            #st.session_state.chunks = db.transcript(st.session_state.chunks, st.session_state.audio_path)
            num_threads=len(st.session_state.chunks)
            try:
                with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
                    args_list = [(chunk,) for chunk in st.session_state.chunks]
                    chunks=executor.map(db.transcript,*zip(*args_list))
            finally:
                executor.shutdown()
            chunks=list(chunks)
            st.session_state.chunks=chunks
            end = time.time()
            if st.session_state.chunks:
                print("Transcription :", round((end-start) / 60, 2), "min\n")
            else:
                raise ValueError("Could not transcribe")
            
        with st.spinner("Analyzing Transcription..."):
            start = time.time()
            st.session_state.chunks = db.modify_transcript(st.session_state.chunks, st.session_state.audio_path)
            end = time.time()
            if st.session_state.chunks:
                print("Transcription Modification :", round((end-start) / 60, 2), "min\n")
            else:
                raise ValueError("Could not modify transcription")

        with st.spinner("Translating..."):
            start = time.time()
            num_threads=len(st.session_state.chunks)
            try:
                with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
                    args_list = [(chunk, st.session_state.source_lang, st.session_state.target_lang) for chunk in st.session_state.chunks]
                    chunks=executor.map(db.translation,*zip(*args_list))
            finally:
                executor.shutdown()
            chunks=list(chunks)
            st.session_state.chunks=chunks
            #st.session_state.chunks = db.translation(st.session_state.chunks, st.session_state.source_lang, st.session_state.target_lang)
            end = time.time()
            print("Translation :", round((end-start) / 60, 2), "min\n")
        
        with st.spinner("Generating Audio..."):
            start = time.time()
            st.session_state.chunks, st.session_state.voice = db.audio_synthesis(st.session_state.chunks)
            end = time.time()
            if os.path.exists("ClonedAudio") and os.listdir("ClonedAudio"):
                print("Audio Generation :", round((end-start) / 60, 2), "min\n")
            else:
                raise FileNotFoundError("Could not synthesize audio")
        
        with st.spinner("Modifying Audio..."):
            start = time.time()
            st.session_state.chunks = db.audio_modification(st.session_state.chunks)
            end = time.time()
            if os.path.exists("ModifiedAudio") and os.listdir("ModifiedAudio"):
                print("Audio Modification :", round((end-start) / 60, 2), "min\n")
            else:
                raise FileNotFoundError("Could not modify audio")

        return None
    except Exception as e:
        st.error(e)
        print(traceback.format_exc())
        return e

def merge():
    with st.spinner("Merging..."):
        try:
            start = time.time()
            st.session_state.output_path = db.non_lipsync(st.session_state.video_path, st.session_state.chunks)
            end = time.time()
            if os.path.exists(st.session_state.output_path):
                print("Merged :", round((end-start) / 60, 2), "min\n")
            else:
                raise FileNotFoundError("No Video File Found")

        except Exception as e:
            st.error(e)
            print(traceback.format_exc())
            return e

def upload():
    try:
        with open(st.session_state.video.name, "wb") as f:
            f.write(st.session_state.video.getbuffer())
        st.video(st.session_state.video.name)
        st.session_state["video_path"] = st.session_state.video.name
        myaudio = AudioSegment.from_file(st.session_state.video.name)
        st.session_state["video_length"] = myaudio.duration_seconds
    except Exception as e:
        st.error(e)
        print(traceback.format_exc())

def generate():
    with st.spinner("Generating Audio..."):
        start = time.time()
        st.session_state.current_chunk = db.audio_synthesis(chunks=[st.session_state.current_chunk],
        stability=st.session_state.stability, similarity_boost=st.session_state.similarity_boost, style=st.session_state.style, boost=st.session_state.boost,
        cloning=False, voice=st.session_state.voice)[0]
        end = time.time()
        if os.path.exists("ClonedAudio") and os.listdir("ClonedAudio"):
            print("Audio Generation :", round((end-start) / 60, 2), "min\n")
        else:
            raise FileNotFoundError("Could not synthesize audio")
    
    with st.spinner("Modifying Audio..."):
        start = time.time()
        st.session_state.current_chunk = db.audio_modification(st.session_state.current_chunk)
        end = time.time()
        if os.path.exists("ModifiedAudio") and os.listdir("ModifiedAudio"):
            print("Audio Modification :", round((end-start) / 60, 2), "min\n")
        else:
            raise FileNotFoundError("Could not modify audio")

    st.session_state.chunks[st.session_state.current] = st.session_state.current_chunk[0]

def get_image_as_base64(path):
    with open(path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

def check():
    if st.session_state.get("source_lang") and st.session_state.get("target_lang"):
        if st.session_state.start_time == 0.0 and st.session_state.end_time == 0.0:
            st.session_state.end_time = st.session_state.video_length
            run()
        else:
            duration = st.session_state.end_time - st.session_state.start_time
            if duration < 60 and duration > 0:
                run()

def main():

    with st.sidebar:
        st.title("Guidelines")
        st.write("This is a beta version of Dublr AI dubbing tool")
        st.write("Only single speaker is supported")
        st.write("Leave start and end time at 0.0, if you wish to dub full video.")
        st.write("Maximum duration of 60 seconds will be dubbed.")
        st.write("After dubbing, select a chunk to view and edit it.")
        st.write("After editing, press Run and then again select chunk from dropdown menu to update it")
        st.write("When you are satisfied with the output, press Merge")
        st.write("In case of any queries, reach us out at info@dublr.ai")
        
    image_path = 'beta.png'
    image_base64 = get_image_as_base64(image_path)
    link = f'<div style="text-align: center;"><a href="https://dublr.ai" target="_blank"><img src="data:image/png;base64,{image_base64}" alt="Dublr" width="200px" style="margin: 0 auto;"></a></div>'
    
    st.markdown(link, unsafe_allow_html=True)
    st.write("")
    st.write("")

    # st.button(label='Home', on_click=remove_keys)

    # with st.expander("State"):
    #     st.write(st.session_state)

    if 'video_path' not in st.session_state:
        with st.container():
            st.file_uploader("Upload Video", on_change=upload, key="video", type=["mp4"])
            
    if "chunks" not in st.session_state and st.session_state.get("video_path"):
        with st.form(key="args_from"):
            st.number_input("Start Time in seconds", key='start_time', min_value=0.0)
            st.number_input("End Time in seconds", key='end_time', max_value=st.session_state.video_length, min_value=0.0)
            st.selectbox(label="Source Language", options=LANG, index=None, key="source_lang")
            st.selectbox(label="Target Language", options=LANG, index=None, key="target_lang")
            st.form_submit_button(label='Submit', on_click=check)
    
    if "output_path" not in st.session_state and st.session_state.get("chunks"):
        st.selectbox(label="Placeholder", options=[i for i in range(len(st.session_state.chunks))], 
        index=None, label_visibility="hidden", placeholder="Select a chunk", key="current")

        if st.session_state.current != None:

            st.session_state.current_chunk = st.session_state.chunks[st.session_state.current]

            with st.container():
                st.write("Text")
                st.session_state.current_chunk["Text"] = st.text_area(label="Placeholder", label_visibility="hidden",
                                                                            value=st.session_state.current_chunk["Text"])

                st.write("Translation")
                st.session_state.current_chunk["Translation"] = st.text_area(label="Placeholder", label_visibility="hidden",
                                                                            value=st.session_state.current_chunk["Translation"])

                st.write("Original Audio")
                st.audio(st.session_state.current_chunk["Path"])

                st.write("Generated Audio")
                st.audio(st.session_state.current_chunk["Generated Path"])

                st.write("Modified Audio")
                st.audio(st.session_state.current_chunk["Modified Path"])

                
                st.slider(label="Stability", min_value=0.0, max_value=1.0, value=0.5, step=0.01, key="stability")
                st.slider(label="Similarity Boost", min_value=0.0, max_value=1.0, value=0.75, step=0.01, key="similarity_boost")
                st.slider(label="Style", min_value=0.0, max_value=1.0, value=0.0, step=0.01, key="style")
                st.checkbox(label="Speaker Boost", key="boost", value=True)
                
                col1, col2 = st.columns(2)
                col1.button(label='Run', on_click=generate)
                col2.button(label="Merge", type="primary", on_click=merge)

    if st.session_state.get("output_path"):
        with st.container():
            with open(st.session_state.output_path, "rb") as file:
                st.video(file.name)
                col1, col2 = st.columns(2)
                col1.button(label="Done", on_click=remove)
                col2.download_button(label="Download", file_name=file.name, data=file, on_click=remove)

if __name__ == "__main__":
    main()
#/home/shaheer/Documents/app/Videos/Starc.mp4