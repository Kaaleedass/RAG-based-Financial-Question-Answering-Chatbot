import os
import pprint
import getpass
import bs4
from bs4 import BeautifulSoup
from urllib.request import Request, urlopen
#from google.colab import userdata

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain_classic.chains import create_retrieval_chain, create_history_aware_retriever
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

import gradio as gr


# ---------------------------------------------------------
# 1. SETUP & DATA LOADING
# ---------------------------------------------------------
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

def get_sitemap(url):
    req = Request(
        url=url,
        headers={"User-Agent": "Mozilla/5.0"}
    )
    response = urlopen(req)
    xml = BeautifulSoup(
        response,
        "lxml-xml",
        from_encoding=response.info().get_param("charset")
    )
    return xml


def get_urls(xml, name=None, data=None, verbose=False):
    urls = []

    for url in xml.find_all("url"):
        loc = url.find("loc")

        if loc and loc.text:
            urls.append(loc.text.strip())

    return urls

url = "https://zerodha.com/varsity/chapter-sitemap2.xml"
xml = get_sitemap(url)
urls = get_urls(xml, verbose=False)

print(f"Found {len(urls)} URLs in the sitemap.")
# Loading just the first 10 for speed; adjust as needed

docs = []

for i, url in enumerate(urls[:10], start=1):
    loader = WebBaseLoader(url)
    loaded_docs = loader.load()
    docs.extend(loaded_docs)

    print(f"Loaded page {i}/10 | Documents: {len(loaded_docs)}")

print(f"\nTotal documents loaded: {len(docs)}")


# ---------------------------------------------------------
# 2. VECTORSTORE & RETRIEVER
# ---------------------------------------------------------
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
retriever = vectorstore.as_retriever()

print(f"Len docs: {len(docs)}, Len splits: {len(splits)}")


# ---------------------------------------------------------
# 3. RAG CHAIN SETUP
# ---------------------------------------------------------
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0) # Updated to newer, cheaper model

system_prompt = (
    "You are a financial assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "If the question is not clear ask follow up questions"
    "\n\n"
    "{context}"
)

# History Aware Retriever
contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)

contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", contextualize_q_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)


rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)


import gradio as gr
from langchain_core.messages import AIMessage, HumanMessage

def predict(message, history):
    """
    message: str
    history: list of dicts in OpenAI-style format when type='messages'
             e.g. [{"role":"user","content":"hi"}, {"role":"assistant","content":"hello"}]
    """
    history_for_llm = []
    for item in history:
        role = item.get("role")
        content = item.get("content", "")
        if role == "user":
            history_for_llm.append(HumanMessage(content=content))
        elif role == "assistant":
            history_for_llm.append(AIMessage(content=content))

    result = rag_chain.invoke({"input": message, "chat_history": history_for_llm})
    return result["answer"]

with gr.Blocks(theme="soft") as demo:
    gr.Markdown("# DocumentQABot")

    chatbot = gr.Chatbot(height=400, allow_tags=False)
    msg = gr.Textbox(
        placeholder="Hi! I am your virtual assistant, how can I help you today?",
        container=False,
        scale=7,
    )

    with gr.Row():
        undo = gr.Button("Delete Previous")
        clear = gr.Button("Clear")

    chat = gr.ChatInterface(
        fn=predict,
        chatbot=chatbot,
        textbox=msg,
        examples=["What is index fund?", "Where to buy stocks?"],
        title=None,  # already using Markdown title
    )

    # Clear chat
    clear.click(lambda: [], None, chatbot)

    # Undo last turn (removes last 2 messages if present: user+assistant)
    def undo_last(history):
        if not history:
            return []
        # In messages format, history is a list of dicts, typically alternating user/assistant
        return history[:-2] if len(history) >= 2 else []

    undo.click(undo_last, chatbot, chatbot)

demo.launch(debug=False)