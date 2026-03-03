# app.py
import streamlit as st
from ingest import create_vectorstore
from chat import ask_question

st.set_page_config(page_title="Finance Report Chat", layout="wide")
st.title("📄 Finance Report Chatbot")

# Step 1: Upload PDF
uploaded_file = st.file_uploader("Upload a PDF financial report", type="pdf")

if uploaded_file is not None:
    with open("temp_report.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success("PDF uploaded successfully! Processing...")

    # Step 2: Create vectorstore from uploaded PDF
    vectorstore = create_vectorstore("temp_report.pdf")
    st.success("Vectorstore created. You can now ask questions!")

    # Step 3: User asks question
    query = st.text_input("Ask a question about the report:")
    if query:
        with st.spinner("Generating answer..."):
            answer = ask_question(vectorstore, query)
        st.markdown("**Answer:**")
        st.write(answer)