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
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    
    # ✅ Correct way for your LangChain version
    docs = retriever.get_relevant_documents_from_query(query)

    context = "\n\n".join([doc.page_content for doc in docs])

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