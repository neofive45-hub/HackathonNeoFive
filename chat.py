# chat.py
import os
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.vectorstores import FAISS

# ---- Read API key from environment variable / Streamlit Secrets ----
api_key = os.getenv("OPENAI_API_KEY")

# ---- Load precomputed vectorstore ----
embeddings = OpenAIEmbeddings(
    base_url="https://genailab.tcs.in",
    model="azure/genailab-maas-text-embedding",
    openai_api_key=api_key
)

# vectorstore = FAISS.load_local("vectorstore", embeddings)

# Now add the safe flag
vectorstore = FAISS.load_local(
    "vectorstore",
    embeddings,
    allow_dangerous_deserialization=True
)

# ---- Function to answer questions ----
def ask_question(query):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    docs = retriever.get_relevant_documents(query)
    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = f"""
    Answer ONLY using the context below.
    If answer not found, say "Not available in document."

    Context:
    {context}

    Question:
    {query}
    """

    llm = ChatOpenAI(
        base_url="https://genailab.tcs.in",
        model="azure/genailab-maas-gpt-4o",
        openai_api_key=api_key
    )

    response = llm.invoke(prompt)
    return response.content