from langchain.document_loaders import SitemapLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import streamlit as st

def decompose_elements(soup, el, class_=None):
    if class_:
        foundings = soup.find_all(el, class_=class_)
    else:
        foundings = soup.find_all(el)

    print(f"Number of {el} elements found: {len(foundings)}")  # 메뉴 개수 출력
    for element in foundings:
        # element.name이 None이면 태그 이름이 없는 것으로 간주되며, 이 경우 element를 문자열로 변환할 때 문제가 발생할 수 있습니다.
        # 하지만, 이 상황이 발생하는 이유는 보통 BeautifulSoup 객체가 잘못된 HTML을 파싱할 때입니다. 따라서, element와 element.name을 확인하는 것은 안전한 코딩 습관입니다.
        if element and element.name:
            print(f"Before decompose: {element}")  # decompose 호출 전 내용 출력
            element.decompose()

    return soup


def parse_page(soup):
    soup = soup.find("body")
    soup = decompose_elements(soup, "style")
    soup = decompose_elements(soup, "div", "x-dropdown")
    soup = decompose_elements(soup, "div", "x-menu")
    soup = decompose_elements(soup, "nav")
    soup = decompose_elements(soup, "section")

    # print(f"HTML after decomposing menus: {soup.prettify()}")

    return (
        str(soup.get_text())
        .replace("\t", " ")
        .replace("\n", " ")
        .replace("\xa0", " ")
    )


@st.cache_data(show_spinner="Loading website...")
def load_website(url):
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000,
        chunk_overlap=100,
    )
    loader = SitemapLoader(
        url,
        filter_urls=[
            # r"^(?!.*blog\/).*",
            # r"^(.*blog\/).*",
            r"^(.*blog\/).*",
            # "https://wordpress.com/tos/",
            # from https://wordpress.com/sitemap-1.xml
        ],
        parsing_function=parse_page,
    )
    loader.requests_per_second = 2
    docs = loader.load_and_split(text_splitter=splitter)
    return docs


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
        placeholder="https://example.com",
    )


if url:
    if ".xml" not in url:
        with st.sidebar:
            st.error("Please write down a Sitemap URL.")
    else:
        docs = load_website(url)
        st.write(docs)