import streamlit as st
import pandas as pd
from gpt4all import GPT4All
import faiss
from nomic import embed
import numpy as np

# ✅ Load your local GPT4All model
# Replace the filename with the path to your downloaded model
MODEL_PATH = r"C:\Users\Minh Tran\AppData\Local\nomic.ai\GPT4All\Meta-Llama-3-8B-Instruct.Q4_0.gguf"
model = GPT4All(MODEL_PATH)

st.title("Chat with Your Dataset (GovGPT)")

# Step 1: Upload dataset
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Preview of dataset:", df.head())

    docs = [str(row) for _, row in df.iterrows()]

    st.write("Generating embeddings...")
    output = [embed.text(texts=[doc], model='nomic-embed-text-v1.5', task_type="search_query",inference_mode='local',) for doc in docs]
   
    emb_matrix = np.array([o['embeddings'][0] for o in output]).astype("float32")
    index = faiss.IndexFlatL2(emb_matrix.shape[1])
    print(emb_matrix.shape)
    index.add(emb_matrix)
    id_to_doc = {i: docs[i] for i in range(len(docs))}
    

    # Step 2: Chat input
    if "history" not in st.session_state:
        st.session_state.history = []

    user_input = st.text_input("Ask a question about your dataset:")

    if user_input:
        query_output = embed.text(
        texts=[user_input],
        model="nomic-embed-text-v1.5",
        task_type="search_document",
        inference_mode="local",
        dimensionality=768
        )

        # Extract embedding vector from the first item
        q_emb = np.array(query_output['embeddings'][0]).astype("float32").reshape(1, -1)  # shape (1, 768)

        # Search top-3 similar rows
        D, I = index.search(q_emb, k=3)
        retrieved_text = " ".join([id_to_doc[i] for i in I[0]])

        with model.chat_session():
            response = model.generate(
                f"""
                You are a helpful assistant that analyzes tabular data.
                Dataset columns: {list(df.columns)}=
                Dataset retrieved rows: {retrieved_text}

                Question: {user_input}
                """
            )

        # Save conversation
        st.session_state.history.append(("User", user_input))
        st.session_state.history.append(("Assistant", response))

    # Display chat history
    for role, text in st.session_state.history:
        if role == "User":
            st.markdown(f"**🧑 {role}:** {text}")
        else:
            st.markdown(f"**🤖 {role}:** {text}")


