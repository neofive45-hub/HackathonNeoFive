import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.vectorstores import FAISS

# Load embeddings
embeddings = OpenAIEmbeddings(
    base_url="https://genailab.tcs.in",
    model="azure_ai/genailab-maas-DeepSeek-R1",
    openai_api_key=os.getenv("OPENAI_API_KEY")
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