from langchain.memory import ConversationSummaryBufferMemory, ConversationBufferWindowMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.document_loaders import UnstructuredFileLoader
from langchain.embeddings import CacheBackedEmbeddings, OpenAIEmbeddings
from langchain.schema import messages_from_dict, messages_to_dict
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.storage import LocalFileStore
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.callbacks.base import BaseCallbackHandler
from datetime import datetime

import streamlit as st
import json

st.set_page_config(
    page_title="DocumentGPT",
    page_icon="📃",
)


class ChatCallbackHandler(BaseCallbackHandler):
    message = ""

    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty()

    def on_llm_end(self, *args, **kwargs):
        save_message(self.message, "ai")

    def on_llm_new_token(self, token, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)


llm = ChatOpenAI(
    temperature=0.1,
    streaming=True,
    callbacks=[
        ChatCallbackHandler(),
    ],
    model_name="gpt-4o-mini"
)


@st.cache_resource
def init_memory():
    # return ConversationSummaryBufferMemory(
    #     llm=llm,
    #     max_token_limit=120,
    #     return_messages=True
    # )
    return ConversationBufferWindowMemory(
        return_messages=True,
        k=4,
    )


def clear_memory():
    memory.clear()


memory = init_memory()


@st.cache_data(show_spinner="Embedding file...")
def embed_file(file):
    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)
    cache_dir = LocalFileStore(f"./.cache/embeddings/{file.name}")
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    # See and download/install: 
    # https://github.com/oschwartz10612/poppler-windows/releases
    # https://github.com/UB-Mannheim/tesseract/wiki
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    embeddings = OpenAIEmbeddings()
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
    vectorstore = FAISS.from_documents(docs, cached_embeddings)
    retriever = vectorstore.as_retriever()
    return retriever


def get_conversation_history_filename():
    now = datetime.now()
    formatted_time = now.strftime("%y%m%d-%H%M%S")
    return f"./.history/{formatted_time}.json"


def save_message(message, role):
    st.session_state["messages"].append({"message": message, "role": role})


def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)


def paint_history():
    for message in st.session_state["messages"]:
        send_message(
            message["message"],
            message["role"],
            save=False,
        )


def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)


prompt = ChatPromptTemplate.from_messages(
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


st.title("DocumentGPT")

st.markdown(
    """
Welcome!
            
Use this chatbot to ask questions to an AI about your files!

Upload your files on the sidebar.
"""
)


def load_memory(_):
    return memory.load_memory_variables({})["history"]


def invoke_chain(question):
    chain = (
        {
            "context": retriever | format_docs,
            "history": RunnableLambda(load_memory),
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
    )
    with st.chat_message("ai"):
        result = chain.invoke(question)
        memory.save_context(
            {"input": question},
            {"output": result.content},
        )


with st.sidebar:
    file = st.file_uploader(
        "Upload a .txt .pdf or .docx file",
        type=["pdf", "txt", "docx"],
    )

if file:
    retriever = embed_file(file)
    send_message("I'm ready! Ask away!", "ai", save=False)
    paint_history()
    message = st.chat_input("Ask anything about your file...")
    if message:
        send_message(message, "human")
        invoke_chain(message)
else:
    st.session_state["messages"] = []
    pathname = get_conversation_history_filename()
    messages = messages_to_dict(load_memory({}))
    if len(messages):
        with open(pathname, "wt") as f:
            json.dump(messages, f, indent=4, ensure_ascii=False)
            clear_memory()