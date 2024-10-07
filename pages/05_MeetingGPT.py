from langchain.storage import LocalFileStore
import streamlit as st
import subprocess
import math
from pydub import AudioSegment
import glob
import openai
import os
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings
from langchain.storage import LocalFileStore


st.set_page_config(
    page_title="MeetingGPT",
    page_icon="💼",
)

from pages.utils.ai_common import llm as llm2, memory, load_memory, format_docs
from pages.utils.chat_handler import ChatCallbackHandler
from pages.utils.chat_handler import send_message, paint_history


llm = ChatOpenAI(
    temperature=0.1,
    model="gpt-4o-mini",
)


splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=800,
    chunk_overlap=100,
)


@st.cache_data()
def embed_file(file_path):
    cache_dir = LocalFileStore(f"./.cache/embeddings/{file.name}")
    loader = TextLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    embeddings = OpenAIEmbeddings()
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
    vectorstore = FAISS.from_documents(docs, cached_embeddings)
    retriever = vectorstore.as_retriever()
    return retriever


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
    with st.status("Loading video...") as status:
        video_content = video.read()
        video_path = f"./.cache/{video.name}"
        video_extension = video.name.split('.')[-1]  # 비디오 파일 확장자 추출
        audio_path = video_path.replace(video_extension, "mp3")  # 동적으로 mp3로 변환
        transcript_path = video_path.replace(video_extension, "txt")  # 동적으로 txt로 변환
        transcript_summary_path = f"./.cache/{video.name.split('.')[0]}_summary.txt"

        with open(video_path, "wb") as f:
            f.write(video_content)
        status.update(label="Extracting audio...")
        extract_audio_from_video(video_path)
        status.update(label="Cutting audio segments...")
        cut_audio_in_chunks(audio_path, 10, chunks_folder)
        status.update(label="Transcribing audio...")
        transcribe_chunks(chunks_folder, transcript_path)

    transcript_tab, summary_tab, qa_tab = st.tabs(
        [
            "Transcript",
            "Summary",
            "Q&A",
        ]
    )

    with transcript_tab:
        with open(transcript_path, "r") as file:
            st.write(file.read())


    with summary_tab:
        start = st.button("Generate summary")
        done_summary = False

        if file_exists(transcript_summary_path):
            done_summary = True

            with open(transcript_summary_path, "rt", encoding="utf-8") as f:
                summary = f.read()
                st.write(summary)

        if start:
            loader = TextLoader(transcript_path)
            docs = loader.load_and_split(text_splitter=splitter)

            first_summary_prompt = ChatPromptTemplate.from_template(
                """
                Write a concise summary of the following:
                "{text}"
                CONCISE SUMMARY:                
            """
            )

            first_summary_chain = first_summary_prompt | llm | StrOutputParser()

            summary = first_summary_chain.invoke(
                {"text": docs[0].page_content},
            )

            refine_prompt = ChatPromptTemplate.from_template(
                """
                Your job is to produce a final summary.
                We have provided an existing summary up to a certain point: {existing_summary}
                We have the opportunity to refine the existing summary (only if needed) with some more context below.
                ------------
                {context}
                ------------
                Given the new context, refine the original summary.
                If the context isn't useful, RETURN the original summary.
                """
            )

            refine_chain = refine_prompt | llm | StrOutputParser()

            with st.status("Summarizing...") as status:
                for i, doc in enumerate(docs[1:]):
                    status.update(label=f"Processing document {i+1}/{len(docs)-1} ")
                    summary = refine_chain.invoke(
                        {
                            "existing_summary": summary,
                            "context": doc.page_content,
                        }
                    )
                    st.write(f"{i+1}. {summary}")
            st.write(summary)
            done_summary = True

            with open(transcript_summary_path, "wt") as f:
                f.write(summary)

    with qa_tab:
        if not done_summary:
            st.markdown("Please generate summary first before asking questions.")
        else:
            retriever = embed_file(transcript_path)

            qna_prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        """
                        Answer the question using ONLY the following context. If you don't know the answer just say you don't know. DON'T make anything up.
                        Context: {context}
                        """,
                    ),
                    MessagesPlaceholder(variable_name="history"),
                    ("human", "{question}"),
                ]
            )
            llm2.streaming = True
            llm2.callbacks = [ChatCallbackHandler()]
            qna_chain = (
                {
                    "context": retriever | RunnableLambda(format_docs),
                    "question": RunnablePassthrough(),
                }
                | RunnablePassthrough.assign(history=load_memory)
                | qna_prompt
                | llm2
            )

            send_message("I'm ready! Ask away!", "ai", save=False)
            paint_history()

            # Try to show chat input box at the bottom, but not working inside tabs
            question = st.chat_input("Enter your questions about the uploaded video...")
            if done_summary and question:
                send_message(question, "human")
                
                with st.chat_message("ai"):
                    result = qna_chain.invoke(question)

                memory.save_context(
                    {"input": question},
                    {"output": result.content.replace("$", "\$")},
                )
else:
    st.session_state["messages"] = []
