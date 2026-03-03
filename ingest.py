# ingest.py
import os
TIKTOKEN_CACHE_DIR = os.path.abspath("tiktoken_cache")
os.environ["TIKTOKEN_CACHE_DIR"] = TIKTOKEN_CACHE_DIR
assert os.path.exists(os.path.join(TIKTOKEN_CACHE_DIR, "9b5ad71b2ce5302211f9c61530b329a4922fc6a4")), "tiktoken cache not found!"

import httpx
from langchain_openai import OpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS

# ---- Proxy & client (cannot change for corporate) ----
proxy_url='http://proxy.tcs.com:8080'
proxies={
    'http://':proxy_url,
    'https://':proxy_url
}

client = httpx.Client(verify=False)

BASE_URL = "https://genailab.tcs.in"

# ---- Load PDF ----
# loader = PyPDFLoader("financial_statement.pdf")  # your PDF file
# documents = loader.load()

# # ---- Split text into chunks ----
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
# docs = text_splitter.split_documents(documents)

# # ---- Create embeddings using corporate model ----
# embeddings = OpenAIEmbeddings(
#     base_url=BASE_URL,
#     model="azure/genailab-maas-text-embedding-3-large",
#     http_client=client,
#     openai_api_key="sk-NKg6tv3sqhMFjX5HxnM6IQ"  # safe locally
# )

# # ---- Store in FAISS vectorstore ----
# vectorstore = FAISS.from_documents(docs, embeddings)
# vectorstore.save_local("vectorstore")

# print("✅ Embeddings created and saved in 'vectorstore/'")

def create_vectorstore(pdf_file_path, embedding_model="azure_ai/genailab-maas-DeepSeek-R1"):
    """
    Load PDF, split text, create embeddings, store in FAISS vectorstore.
    Returns the FAISS vectorstore object.
    """
    # Load PDF
    loader = PyPDFLoader(pdf_file_path)
    documents = loader.load()
    
    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    docs = text_splitter.split_documents(documents)

    # Create embeddings
    embeddings = OpenAIEmbeddings(
    base_url="https://genailab.tcs.in",
    model="genailab-maas-text-embedding-3-large",
    openai_api_key="sk-NKg6tv3sqhMFjX5HxnM6IQ"
    )

    # Create FAISS vectorstore in memory
    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore