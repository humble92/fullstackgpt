import aiohttp
import asyncio
import gzip
from io import BytesIO
from bs4 import BeautifulSoup
import streamlit as st

@st.cache_data(show_spinner="Loading website...")
def load_website(url):
    return asyncio.run(fetch_and_parse_sitemap(url))


async def fetch_and_parse_sitemap(url):
    xml_content = await fetch_gzip(url)
    soup = BeautifulSoup(xml_content, 'xml')
    return soup


async def fetch_gzip(url):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            if response.status != 200:
                response.raise_for_status()
            compressed_data = await response.read()
            with gzip.GzipFile(fileobj=BytesIO(compressed_data)) as f:
                return f.read().decode('utf-8')


st.set_page_config(
    page_title="SiteGPT",
    page_icon="🖥️",
)

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
        placeholder="https://example.com/sitemap.xml.gz",
    )

if url:
    if ".xml.gz" not in url:
        with st.sidebar:
            st.error("Please write down a Sitemap URL ending with .xml.gz.")
    else:
        try:
            soup = load_website(url)
            st.write(soup.prettify())
        except Exception as e:
            st.error(f"Error fetching or parsing the sitemap: {e}")