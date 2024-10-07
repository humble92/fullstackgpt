from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferWindowMemory
from langchain.schema import Document
import streamlit as st


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
    #     memory_key="history",
    #     return_messages=True,
    # )
    return ConversationBufferWindowMemory(
        memory_key="history",
        return_messages=True,
        k=4,
    )
memory = init_memory()


def load_memory(_):
    return memory.load_memory_variables({})["history"]


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
            score_threshold=0.75,
        )
        candidate, score = (
            found_docs[0][0].page_content.split("\n")[1],
            found_docs[0][1],
        )
        print(found_docs)
        print(f"Note: found docs from chat history with score {score}")
        return candidate.replace("output:", "")
    except IndexError:
        print(f"Note: not found in the vector store")
        return None


def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)
