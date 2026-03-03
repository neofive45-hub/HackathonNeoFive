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
    # Create retriever from vectorstore
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    
    # Correct way to get documents
    docs = retriever.retrieve(query)  # <-- use retrieve(), not get_relevant_documents()
    
    # Combine context
    context = "\n\n".join([doc.page_content for doc in docs])

    # LLM for answering
    llm = ChatOpenAI(
        base_url="https://genailab.tcs.in",
        model="azure/genailab-maas-gpt-4o",
        openai_api_key=os.getenv("OPENAI_API_KEY")
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