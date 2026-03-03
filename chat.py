from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.vectorstores import FAISS
import os

# ---- Load vectorstore with embeddings ----
embeddings = OpenAIEmbeddings(
    base_url="https://genailab.tcs.in",
    model="azure_ai/genailab-maas-DeepSeek-R1",  # embedding model
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

# Allow loading local pickle (safe because you created it)
vectorstore = FAISS.load_local(
    "vectorstore",
    embeddings,
    allow_dangerous_deserialization=True
)

def ask_question(query):
    # Get relevant documents directly from the vectorstore
    docs = vectorstore.get_relevant_documents(query)  # Works in your version

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