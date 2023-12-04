import streamlit as st
import dublr as db
import time
import warnings
import os
import requests
from dotenv import load_dotenv
load_dotenv()

warnings.filterwarnings('ignore')

def remove():
    url = f"https://api.elevenlabs.io/v1/voices/{st.session_state.voice.voice_id}"
    headers = {'xi-api-key': os.getenv("11LABs"),}
    response = requests.request("DELETE", url, headers=headers)

    if response.status_code != 200:
        raise LookupError(f"Error: {response.status_code} - {response.text}")

    for key in st.session_state.keys():
        del st.session_state[key]

def run():
    db.remove_data()
    try:
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
            st.session_state.chunks = db.transcript(st.session_state.chunks, st.session_state.audio_path)
            end = time.time()
            if st.session_state.chunks:
                print("Transcription :", round((end-start) / 60, 2), "min\n")
            else:
                raise ValueError("Could not transcribe")

        with st.spinner("Translating..."):
            start = time.time()
            st.session_state.chunks = db.translation(st.session_state.chunks, st.session_state.source_lang, st.session_state.target_lang)
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
        st.exception(e)
        print(str(e))
        return e

def merge():
    try:
        start = time.time()
        db.chunks_to_srt(st.session_state.chunks)
        end = time.time()
        if os.path.exists('Data/sub.srt'):
            print("Subtitles :", round((end-start) / 60, 2), "min\n")
        else:
            raise FileNotFoundError("Could not Generate SRT")

        start = time.time()
        st.session_state.output_path = db.non_lipsync(st.session_state.video_path, st.session_state.chunks)
        end = time.time()
        if os.path.exists(st.session_state.output_path):
            print("Merged :", round((end-start) / 60, 2), "min\n")
        else:
            raise FileNotFoundError("No Video File Found")

    except Exception as e:
        st.exception(e)
        print(str(e))
        return e

def main():
    st.write(st.session_state)
    
    if 'video_path' not in st.session_state:
        with st.form(key='video_form'):
            video = st.file_uploader("Upload Video")
            submit = st.form_submit_button(label='Load Video')
            if submit and video:
                try:
                    with open(video.name, "wb") as f:
                        f.write(video.getbuffer())
                    st.video(video.name)
                    st.session_state["video_path"] = video.name
                except Exception as e:
                    st.exception(e)

    if "chunks" not in st.session_state and st.session_state.get("video_path"):
        with st.form(key="args_from"):
            st.number_input("Start Time in seconds", key='start_time')
            st.number_input("End Time in seconds", key='end_time', min_value=st.session_state.start_time)
            st.text_input("Source Language", key='source_lang')
            st.text_input("Target Language", key='target_lang')
            st.form_submit_button(label='Submit', on_click=run)
    
    if "output_path" not in st.session_state and st.session_state.get("chunks"):
        st.selectbox(label="Placeholder", options=[i for i in range(len(st.session_state.chunks))], 
        index=None, label_visibility="hidden", placeholder="Select a chunk", key="current")

        if st.session_state.current != None:

            st.session_state.current_chunk = st.session_state.chunks[st.session_state.current]

            with st.form(key='change_chunk'):
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
                submit = col1.form_submit_button(label='Run')

                if submit:
                    
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
                col2.form_submit_button(label="Merge", type="primary", on_click=merge)

    if st.session_state.get("output_path"):
        st.video(st.session_state.output_path)
        st.button(label="Done", on_click=remove)
if __name__ == "__main__":
    main()
#/home/shaheer/Documents/app/Videos/Starc.mp4