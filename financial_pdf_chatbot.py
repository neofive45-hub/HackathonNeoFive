import os
TIKTOKEN_CACHE_DIR = os.path.abspath("tiktoken_cache")
os.environ["TIKTOKEN_CACHE_DIR"] = TIKTOKEN_CACHE_DIR
assert os.path.exists(os.path.join(TIKTOKEN_CACHE_DIR, "9b5ad71b2ce5302211f9c61530b329a4922fc6a4")), "tiktoken cache not found!"

import streamlit as st
import httpx
import tempfile

from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA

# -----------------------------
# Proxy Setup 
# -----------------------------
proxy_url='http://proxy.tcs.com:8080'
proxies={
    'http://':proxy_url,
    'https://':proxy_url
}

client = httpx.Client(verify=False)

BASE_URL = "https://genailab.tcs.in"

# -----------------------------
# Model List (from your code)
# -----------------------------
model_list = [
    'azure/genailab-maas-gpt-4o',
    'azure/genailab-maas-gpt-4o-mini',
    'azure_ai/genailab-maas-DeepSeek-R1',
    'azure_ai/genailab-maas-DeepSeek-V3-0324',
    'azure_ai/genailab-maas-Llama-3.3-70B-Instruct'
]

selected_model = st.sidebar.selectbox("Select Model", model_list)

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("📊 Financial PDF Chatbot")

st.write("Upload a financial PDF and ask questions about it.")

uploaded_file = st.file_uploader("Upload PDF", type="pdf")

if uploaded_file:

    # Save PDF temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        pdf_path = tmp.name

    st.success("PDF Uploaded Successfully")

    # -----------------------------
    # Load PDF
    # -----------------------------
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    # -----------------------------
    # Split Text
    # -----------------------------
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )

    chunks = text_splitter.split_documents(documents)

    st.write(f"Document split into {len(chunks)} chunks")

    # -----------------------------
    # Embeddings
    # -----------------------------
    embeddings = HuggingFaceEmbeddings(
        model_name="genailab-maas-text-embedding-3-large"
    )

    # -----------------------------
    # Vector Database
    # -----------------------------
    vectordb = Chroma.from_documents(
        chunks,
        embeddings
    )

    retriever = vectordb.as_retriever(search_kwargs={"k":3})

    # -----------------------------
    # LLM (Your Template)
    # -----------------------------
    llm = ChatOpenAI(
        base_url="https://genailab.tcs.in",
        model="azure/genailab-maas-gpt-4o",
        openai_api_key="sk-NKg6tv3sqhMFjX5HxnM6IQ",
        http_client=client
    )

    # -----------------------------
    # Retrieval QA
    # -----------------------------
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever
    )

    st.divider()

    question = st.text_input("Ask a question about the document")

    if question:

        with st.spinner("Searching document..."):

            response = qa_chain.invoke({"query": question})

        st.subheader("Answer")

        st.write(response["result"])