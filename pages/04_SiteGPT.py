import streamlit as st

st.set_page_config(
    page_title="SiteGPT",
    page_icon="🖥️",
)

from langchain.document_loaders import SitemapLoader
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chat_models import ChatOpenAI

from pages.utils.chat_handler import ChatCallbackHandler
from pages.utils.ai_common import llm, memory, load_memory, find_history
from pages.utils.chat_handler import send_message, paint_history
from pages.utils.url_loader import decompose_elements

import json

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

            Cite the sources of the answers with meta data(e.g. last modified date). If the source is the same, just cite it once.

            Answers: {answers}
            """,
        ),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ]
)


def choose_answer(inputs):
    answers = inputs["answers"]
    question = inputs["question"]
    history = inputs["history"]

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
            "history": history,
        }
    )


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


with st.sidebar:
    url = st.text_input(
        "Write down a URL",
        placeholder="https://example.com",
    )


cache_prompt = ChatPromptTemplate.from_template(
    """
    Find out an answer of a question from cache, and return the result as follows:

    When the answer is found: "found": true, "answer": "...."
    When not found: "found": false

    cache:
    {history}

    question:
    {question}
"""
)

retrieve_from_cache = {
    "name": "retrieve_from_cache",
    "description": "Retrieve an answer of a question from cache",
    "parameters": {
        "type": "object",
        "properties": {
            "found": {
                "type": "boolean",
                "description": "if the answer of the question is found from cache, this is true",
            },
            "answer": {
                "type": "string",
                "description": "the answer of the question",
            },
        },
    },
    "required": ["found"],
}

cache_llm = ChatOpenAI(
    temperature=0.1,
    model="gpt-4o-mini",
).bind(
    functions=[
        retrieve_from_cache,
    ],
    function_call={"name": "retrieve_from_cache"},
)


def find_answer_or_invoke_chain(query):
    with st.spinner("checking cache.."):
        condensed = "\n".join(
            f"{message['role']}: {message['msg']}"
            for message in st.session_state["messages"]
        )
        cache_chain = cache_prompt | cache_llm
        cache_result = json.loads(
            cache_chain.invoke(
                {
                    "history": condensed,
                    "question": query,
                }
            ).additional_kwargs["function_call"]["arguments"]
        )
        print(cache_result)

    if cache_result["found"]:
        result = cache_result["answer"]
        send_message(result, "ai")
    else:
        with st.chat_message("ai"):
            chain = (
                {
                    "docs": retriever,
                    "question": RunnablePassthrough(),
                }
                | RunnableLambda(get_answers)
                | RunnablePassthrough.assign(history=load_memory)
                | RunnableLambda(choose_answer)
            )
            result = chain.invoke(query)

        memory.save_context(
            {"input": query},
            {"output": result.content.replace("$", "\$")},
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
                find_answer_or_invoke_chain(query)
else:
    st.session_state["messages"] = []