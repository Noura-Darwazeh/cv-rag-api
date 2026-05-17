import streamlit as st
import requests

st.title("🧠 CV RAG Assistant")

question = st.text_input("Ask a question:")

show_context = st.checkbox("Show context")

if st.button("Ask"):
    res = requests.post(
        "http://127.0.0.1:8000/ask",
        json={
            "question": question,
            "show_context": show_context
        }
    )

    if res.status_code == 200:
        data = res.json()

        st.subheader("Answer")
        st.write(data["answer"])

        if show_context:
            st.subheader("Context")
            st.text(data["context"])
    else:
        st.error(res.text)