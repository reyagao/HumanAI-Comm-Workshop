import streamlit as st
from openai import OpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Fixed configuration (MUST match the indexing step)
COLLECTION_NAME = "doc_index"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHROMA_PERSIST_DIR = "./chroma_db"

TOP_K = 4  # Number of chunks to retrieve per query

# Streamlit UI
st.title("📄Course Reading Chatbot")
st.caption(
    "Retrieval-Augmented Generation using a local Chroma vector store "
    "and Qwen."
)

# Initialize Qwen client
client = OpenAI(
    api_key=st.secrets["DASHSCOPE_API_KEY"],
    base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
)

if "model" not in st.session_state:
    st.session_state["model"] = "qwen-plus"

# Load embeddings + Chroma vector store (initialize once)
def load_retriever():
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        multi_process=False,
    )

    db = Chroma(
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_PERSIST_DIR,
    )
    return db


db = load_retriever()

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# User input
if prompt := st.chat_input("Your message"):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    # Retrieval: similarity search in Chroma
    docs = db.similarity_search(prompt, k=TOP_K)

    context_blocks = []
    for i, doc in enumerate(docs, 1):
        context_blocks.append(f"[Document {i}]\n{doc.page_content}")

    context = "\n\n".join(context_blocks)

    # Build the RAG prompt
    rag_prompt = f"""
You are a helpful assistant. Answer the question using ONLY the information
from the documents below. If the answer is not contained in the documents,
say you do not know.

Documents:
{context}

Question:
{prompt}

Answer:
""".strip()

    # Call Qwen
    with st.chat_message("assistant"):
        stream = client.chat.completions.create(
            model=st.session_state["model"],
            messages=[{"role": "user", "content": rag_prompt}],
            stream=True,
        )
        response = st.write_stream(stream)

    # Save the assistant response
    st.session_state.messages.append({"role": "assistant", "content": response})
