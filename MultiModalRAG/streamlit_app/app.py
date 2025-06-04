# streamlit_app/app.py

import streamlit as st
import requests

API_URL = "http://localhost:8000"

st.set_page_config(page_title="Multimodal RAG", layout="wide")
st.title("🧠 Multimodal RAG App")

tabs = st.tabs(["📤 Upload", "🔍 Search", "💬 Chat"])


# 1. UPLOAD TAB
with tabs[0]:
    st.header("Upload Document or Image")

    uploaded_file = st.file_uploader(
        "Choose a file", type=["pdf", "docx", "md", "xlsx", "png", "jpg", "jpeg"]
    )
    filetype = st.selectbox(
        "Select file type", ["pdf", "docx", "markdown", "excel", "image"]
    )

    if st.button("Upload"):
        if uploaded_file:
            files = {"file": (uploaded_file.name, uploaded_file)}
            data = {"filetype": filetype}

            res = requests.post(f"{API_URL}/upload/", files=files, data=data)
            st.success(res.json().get("message"))
        else:
            st.warning("Please upload a file first.")

    if st.button("🚀 Run Ingestion"):
        with st.spinner("Embedding and storing..."):
            res = requests.post(f"{API_URL}/ingest/")
            st.success(res.json().get("status"))


# 2. SEARCH TAB
with tabs[1]:
    st.header("Semantic Search")

    query = st.text_input("Enter your query")

    if st.button("Search"):
        res = requests.post(f"{API_URL}/search/", json={"query": query, "top_k": 5})
        results = res.json().get("results", [])

        if results:
            for i, result in enumerate(results, 1):
                st.markdown(
                    f"**Result {i}:** {result['metadata'].get('source_file', 'N/A')}"
                )
                st.write(result["metadata"].get("text", ""))
                st.code(result["metadata"])
        else:
            st.warning("No results found.")


# 3. CHAT TAB
with tabs[2]:
    st.header("Conversational Chat with RAG")
    import uuid

    # Initialize session
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.chat_memory = []

    # Upload inside chat
    with st.expander("📎 Upload new file here (doc/image) to enrich context"):
        uploaded_file = st.file_uploader(
            "Upload a file for contextual enrichment",
            type=["pdf", "docx", "md", "xlsx", "png", "jpg", "jpeg"],
        )
        filetype = st.selectbox(
            "Select file type for upload", ["pdf", "docx", "markdown", "excel", "image"]
        )

        if st.button("Upload File"):
            if uploaded_file:
                files = {"file": (uploaded_file.name, uploaded_file)}
                data = {"filetype": filetype}

                res = requests.post(f"{API_URL}/upload/", files=files, data=data)
                if res.ok:
                    st.success(f"{uploaded_file.name} uploaded successfully.")
                    st.session_state.chat_memory.append(
                        {
                            "role": "assistant",
                            "text": f"✅ File `{uploaded_file.name}` uploaded and ready for ingestion.",
                        }
                    )

                    # Auto-ingest
                    ingest_response = requests.post(f"{API_URL}/ingest/")
                    if ingest_response.ok:
                        st.success("Vector store updated with new content.")
                        st.session_state.chat_memory.append(
                            {
                                "role": "assistant",
                                "text": f"🧠 Ingested `{uploaded_file.name}` into the knowledge base.",
                            }
                        )
                else:
                    st.error("Upload failed. Check backend.")
            else:
                st.warning("No file selected.")

    # Chat input
    user_input = st.chat_input("Ask a question...")

    if user_input:
        st.session_state.chat_memory.append({"role": "user", "text": user_input})

        with st.spinner("Thinking..."):
            res = requests.post(f"{API_URL}/chat/", json={"question": user_input})
            answer = res.json().get("answer", "No answer.")
            st.session_state.chat_memory.append({"role": "assistant", "text": answer})

    # Display chat history
    for msg in st.session_state.chat_memory:
        if msg["role"] == "user":
            st.chat_message("user").write(msg["text"])
        else:
            # Parse possible image preview
            if "image_path=" in msg["text"]:
                text, image_path = msg["text"].split("image_path=")
                st.chat_message("assistant").write(text.strip())
                st.image(image_path.strip(), width=300, caption="Embedded image")
            else:
                st.chat_message("assistant").write(msg["text"])


"""
API Requirements Recap
Ensure your FastAPI backend:
Is running at http://localhost:8000
Has endpoints:
POST /upload/
POST /ingest/
POST /search/
POST /chat/
"""
