import os
from langchain_openai import ChatOpenAI
from langchain.vectorstores import FAISS

# Assuming vectorstore is already loaded
# vectorstore = FAISS.load_local("vectorstore", embeddings, allow_dangerous_deserialization=True)

def ask_question(query):
    # Create retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    
    # Call retriever as a function to get docs
    docs = retriever(query)  # <-- This is the correct usage now

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