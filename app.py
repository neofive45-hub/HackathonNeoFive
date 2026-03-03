# app.py
import streamlit as st
from chat import ask_question

st.set_page_config(page_title="Financial PDF Chat", page_icon="💰")
st.title("💰 Financial PDF Chatbot")

# User input
query = st.text_input("Ask a question about the financial statement:")

if query:
    with st.spinner("Searching for answers..."):
        answer = ask_question(query)
    st.markdown("**Answer:**")
    st.write(answer)