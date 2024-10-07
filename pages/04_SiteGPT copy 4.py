from langchain.document_loaders import SitemapLoader
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.callbacks import StreamingStdOutCallbackHandler
import streamlit as st

llm = ChatOpenAI(
    temperature=0.1,
    model="gpt-4o-mini",
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()],
)


answers_prompt = ChatPromptTemplate.from_template(
    """
    Using ONLY the following context answer the user's question. If you can't just say you don't know, don't make anything up.
                                                  
    Then, give a score to the answer between 0 and 5.

    If the answer answers the user question, the score should be high, else it should be low.

    Make sure to always include the answer's score even if it's 0.

    Context: {context}
                                                  
    Examples:
                                                  
    Question: How far away is the moon?
    Answer: The moon is 384,400 km away.
    Score: 5
                                                  
    Question: How far away is the sun?
    Answer: I don't know
    Score: 0
                                                  
    Your turn!

    Question: {question}
"""
)


def get_answers(inputs):
    docs = inputs["docs"]
    question = inputs["question"]
    answers_chain = answers_prompt | llm
    answers = []
    for doc in docs:
        result = answers_chain.invoke(
            {"question": question, "context": doc.page_content}
        )
        answers.append(result.content)
    st.write(answers)


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
    vector_store = FAISS.from_documents(docs, OpenAIEmbeddings())
    return vector_store.as_retriever()


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
        retriever = load_website(url)

        chain = {
            "docs": retriever,
            "question": RunnablePassthrough(),
        } | RunnableLambda(get_answers)

        chain.invoke("How to monetize using blog?")