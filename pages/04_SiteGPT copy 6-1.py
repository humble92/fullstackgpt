from langchain.document_loaders import SitemapLoader
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferWindowMemory
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import Document
import streamlit as st


st.set_page_config(
    page_title="SiteGPT",
    page_icon="🖥️",
)


llm = ChatOpenAI(
    temperature=0.1,
    model="gpt-4o-mini",
)


@st.cache_resource
def init_memory():
    # NotImplementedError: get_num_tokens_from_messages() is not presently implemented for model cl100k_base.
    # See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens.
    # memory = ConversationSummaryBufferMemory(
    #     llm=llm,
    #     max_token_limit=150,
    #     memory_key="chat_history",
    #     return_messages=True,
    # )
    return ConversationBufferWindowMemory(
        memory_key="chat_history",
        return_messages=True,
        k=4,
    )

memory = init_memory()


def load_memory(_):
    return memory.load_memory_variables({})["chat_history"]


answers_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Using ONLY the following context answer the user's question. If you can't just say you don't know, don't make anything up.
                                                        
            Then, give a score to the answer between 0 and 5.
            If the answer answers the user question the score should be high, else it should be low.
            Make sure to always include the answer's score even if it's 0.
            Context: {context}
                                                        
            Examples:
                                                        
            Question: How far away is the moon?
            Answer: The moon is 384,400 km away.
            Score: 5
                                                        
            Question: How far away is the sun?
            Answer: I don't know
            Score: 0    
            """,
        ),
        ("human", "{question}"),
    ]
)


def get_answers(inputs):
    docs = inputs["docs"]
    question = inputs["question"]

    llm.streaming = False
    llm.callbacks = None
    answers_chain = answers_prompt | llm
    # answers = []
    # for doc in docs:
    #     result = answers_chain.invoke(
    #         {"question": question, "context": doc.page_content}
    #     )
    #     answers.append(result.content)
    return {
        "question": question,
        "answers": [
            {
                "answer": answers_chain.invoke(
                    {
                        "question": question, 
                        "context": doc.page_content,
                    }
                ).content,
                "source": doc.metadata.get("source", ""),
                "date": doc.metadata.get("lastmod", ""),
            }
            for doc in docs
        ],
    }


choose_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Use ONLY the following pre-existing answers to answer the user's question.

            Use the answers that have the highest score (more helpful) and favor the most recent ones.

            Cite the sources of the answers. If the source is the same, just cite it once.

            Answers: {answers}
            """,
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ]
)


def choose_answer(inputs):
    answers = inputs["answers"]
    question = inputs["question"]
    chat_history = inputs["chat_history"]

    llm.streaming = True
    llm.callbacks = [ChatCallbackHandler()]
    choose_chain = choose_prompt | llm
    condensed = "\n\n".join(
        f"{answer['answer']}\nSource:{answer['source']}\nDate:{answer['date']}\n"
        for answer in answers
    )
    return choose_chain.invoke(
        {
            "question": question,
            "answers": condensed,
            "chat_history": chat_history,
        }
    )


def decompose_elements(soup, el, class_=None):
    if class_:
        foundings = soup.find_all(el, class_=class_)
    else:
        foundings = soup.find_all(el)

    # print(f"Number of {el} elements found: {len(foundings)}")  # 메뉴 개수 출력
    for element in foundings:
        # element.name이 None이면 태그 이름이 없는 것으로 간주되며, 이 경우 element를 문자열로 변환할 때 문제가 발생할 수 있습니다.
        # 하지만, 이 상황이 발생하는 이유는 보통 BeautifulSoup 객체가 잘못된 HTML을 파싱할 때입니다. 따라서, element와 element.name을 확인하는 것은 안전한 코딩 습관입니다.
        if element and element.name:
            # print(f"Before decompose: {element}")  # decompose 호출 전 내용 출력
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


st.markdown(
    """
    # SiteGPT
            
    Ask questions about the content of a website.
            
    Start by writing the URL of the website on the sidebar.
"""
)


def save_message(msg, role):
    st.session_state["messages"].append({"msg": msg, "role": role})


def send_message(msg, role, save=True):
    with st.chat_message(role):
        st.markdown(msg)
    if save:
        save_message(msg, role)


def paint_history():
    for msg in st.session_state["messages"]:
        send_message(msg["msg"], msg["role"], False)


class ChatCallbackHandler(BaseCallbackHandler):
    message = ""

    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty()

    def on_llm_end(self, *args, **kwargs):
        save_message(self.message, "ai")

    def on_llm_new_token(self, token, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)


def find_history(query):
    histories = load_memory(None)
    temp = [
        {
            "input": histories[idx * 2].content,
            "output": histories[idx * 2 + 1].content,
        }
        for idx in range(len(histories) // 2)
    ]

    docs = [
        Document(page_content=f"input:{item['input']}\noutput:{item['output']}")
        for item in temp
    ]

    try:
        vector_store = FAISS.from_documents(docs, OpenAIEmbeddings())
        found_docs = vector_store.similarity_search_with_relevance_scores(
            query,
            k=1,
            score_threshold=0.1,
        )
        candidate, score = (
            found_docs[0][0].page_content.split("\n")[1],
            found_docs[0][1],
        )
        print(found_docs)
        print(f"Note: found docs from chat history with score {score}")
        return candidate.replace("output:", "")
    except IndexError:
        return None


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
        send_message("I'm ready! Ask away!", "ai", False)
        paint_history()

        query = st.chat_input("Ask a question to the website.")
        if query:
            send_message(query, "human")

            found = find_history(query)
            if found:
                send_message(found, "ai")

            else:
                with st.chat_message("ai"):
                    chain = (
                        {
                            "docs": retriever,
                            "question": RunnablePassthrough(),
                        }
                        | RunnableLambda(get_answers)
                        | RunnablePassthrough.assign(chat_history=load_memory)
                        | RunnableLambda(choose_answer)
                    )
                    result = chain.invoke(query)

                memory.save_context(
                    {"input": query},
                    {"output": result.content.replace("$", "\$")},
                )
else:
    st.session_state["messages"] = []