import streamlit as st
import subprocess
import math
from pydub import AudioSegment
import glob
import openai
import os


def file_exists(file_path):
    return os.path.exists(file_path)


@st.cache_data()
def transcribe_chunks(chunk_folder, destination):
    if file_exists(destination):
        return
    
    files = glob.glob(f"{chunk_folder}/*.mp3")
    files.sort()
    for file in files:
        with open(file, "rb") as audio_file, open(destination, "a") as text_file:
            transcript = openai.Audio.transcribe(
                "whisper-1",
                audio_file,
            )
            text_file.write(transcript["text"])


@st.cache_data()
def extract_audio_from_video(video_path):
    video_extension = video_path.split('.')[-1]  # 비디오 파일 확장자 추출
    audio_path = video_path.replace(video_extension, "mp3")

    if file_exists(audio_path):
        return

    command = [
        "ffmpeg",
        "-y",
        "-i",
        video_path,
        "-vn",
        audio_path,
    ]
    subprocess.run(command)


@st.cache_data()
def cut_audio_in_chunks(audio_path, chunk_size, chunks_folder):
    track = AudioSegment.from_mp3(audio_path)
    chunk_len = chunk_size * 60 * 1000
    chunks = math.ceil(len(track) / chunk_len)

    if file_exists(f"./{chunks_folder}/chunk_{chunks-1}.mp3"):
        return
    
    for i in range(chunks):
        start_time = i * chunk_len
        end_time = (i + 1) * chunk_len
        chunk = track[start_time:end_time]
        chunk.export(
            f"./{chunks_folder}/chunk_{i}.mp3",
            format="mp3",
        )


st.set_page_config(
    page_title="MeetingGPT",
    page_icon="💼",
)

st.markdown(
    """
# MeetingGPT
            
Welcome to MeetingGPT, upload a video and I will give you a transcript, a summary and a chat bot to ask any questions about it.

Get started by uploading a video file in the sidebar.
"""
)

with st.sidebar:
    video = st.file_uploader(
        "Video",
        type=["mp4", "avi", "mkv", "mov"],
    )

if video:
    chunks_folder = "./.cache/chunks"
    with st.status("Loading video..."):
        video_content = video.read()
        video_path = f"./.cache/{video.name}"
        video_extension = video.name.split('.')[-1]  # 비디오 파일 확장자 추출
        audio_path = video_path.replace(video_extension, "mp3")  # 동적으로 mp3로 변환
        transcript_path = video_path.replace(video_extension, "txt")  # 동적으로 txt로 변환
        with open(video_path, "wb") as f:
            f.write(video_content)
    with st.status("Extracting audio..."):
        extract_audio_from_video(video_path)
    with st.status("Cutting audio segments..."):
        cut_audio_in_chunks(audio_path, 10, chunks_folder)
    with st.status("Transcribing audio..."):
        transcribe_chunks(chunks_folder, transcript_path)