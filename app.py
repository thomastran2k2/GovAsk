import streamlit as st
import pandas as pd
from gpt4all import GPT4All

# ✅ Load your local GPT4All model
# Replace the filename with the path to your downloaded model
MODEL_PATH = r"C:\Users\Minh Tran\AppData\Local\nomic.ai\GPT4All\Meta-Llama-3-8B-Instruct.Q4_0.gguf"
model = GPT4All(MODEL_PATH)

st.title("Chat with Your Dataset (GPT4All 🤖)")

# Step 1: Upload dataset
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Preview of dataset:", df.head())

    # Step 2: Chat input
    if "history" not in st.session_state:
        st.session_state.history = []

    user_input = st.text_input("Ask a question about your dataset:")

    if user_input:
        with model.chat_session():
            response = model.generate(
                f"""
                You are a helpful assistant that analyzes tabular data.
                Dataset columns: {list(df.columns)}
                First 3 rows: {df.head(3).to_dict()}

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


