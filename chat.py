from langchain_openai import ChatOpenAI, OpenAIEmbeddings
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

# Load FAISS vectorstore
vectorstore = FAISS.load_local(
    "vectorstore",
    embeddings,  # already defined
    allow_dangerous_deserialization=True
)

def ask_question(query):
    # Create a retriever object
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    # Use the retriever's .get_relevant_documents() method
    docs = retriever.get_relevant_documents(query)  # ✅ works in your version

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