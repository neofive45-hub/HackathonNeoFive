import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.vectorstores import FAISS

TIKTOKEN_CACHE_DIR = os.path.abspath("tiktoken_cache")
os.environ["TIKTOKEN_CACHE_DIR"] = TIKTOKEN_CACHE_DIR
assert os.path.exists(os.path.join(TIKTOKEN_CACHE_DIR, "9b5ad71b2ce5302211f9c61530b329a4922fc6a4")), "tiktoken cache not found!"


# Load embeddings
embeddings = OpenAIEmbeddings(
    base_url="https://genailab.tcs.in",
    model="azure/genailab-maas-text-embedding-3-large",
    openai_api_key="sk-NKg6tv3sqhMFjX5HxnM6IQ"
)

# Load vectorstore safely
vectorstore = FAISS.load_local(
    "vectorstore",
    embeddings,
    allow_dangerous_deserialization=True
)

def ask_question(query):
    # Search for top 4 relevant chunks
    docs = vectorstore.similarity_search(query, k=4)  # ✅ works in all versions

    # Combine context
    context = "\n\n".join([doc.page_content for doc in docs])

    # LLM for answering
    llm = ChatOpenAI(
        base_url="https://genailab.tcs.in",
        model="azure/genailab-maas-gpt-4o",
        openai_api_key="sk-NKg6tv3sqhMFjX5HxnM6IQ"
    )

    prompt = f"""
    Answer ONLY using the context below.
    If answer not found, say "Not available in document."

    Context:
    {context}

    Question:
    {query}
    """

    response = llm.invoke(prompt)
    return response.content