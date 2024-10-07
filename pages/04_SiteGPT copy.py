import asyncio
import streamlit as st
from langchain.document_loaders import AsyncChromiumLoader
from langchain.document_transformers import Html2TextTransformer

# Windows 환경에서 asyncio 이벤트 루프 정책 설정
if __name__ == "__main__":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

st.set_page_config(
    page_title="SiteGPT",
    page_icon="🖥️",
)

html2text_transformer = Html2TextTransformer()

st.markdown(
    """
    # SiteGPT
            
    Ask questions about the content of a website.
            
    Start by writing the URL of the website on the sidebar.
"""
)


with st.sidebar:
    url = st.text_input(
        "Write down a URL",
        placeholder="https://example.com",
    )


if url:
    loader = AsyncChromiumLoader([url])
    docs = loader.load()
    transformed = html2text_transformer.transform_documents(docs)
    st.write(docs)