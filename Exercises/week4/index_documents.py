"""Index source documents and persist in vector embedding database."""
import os
from transformers import AutoTokenizer
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

SOURCE_DOCUMENTS = ["SourceDocument/Guzman 2018.pdf"]
COLLECTION_NAME = "doc_index"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHROMA_PERSIST_DIR = "./chroma_db"


def main():
    print("Ingesting...")
    all_docs = ingest_docs(SOURCE_DOCUMENTS)
    print("Persisting...")
    db = generate_embed_index(all_docs)
    print("Done.")


def ingest_docs(source_documents):
    all_docs = []
    for source_doc in source_documents:
        print(source_doc)
        docs = pdf_to_chunks(source_doc)
        all_docs = all_docs + docs
    return all_docs


def pdf_to_chunks(pdf_file):
    # Use the tokenizer from the embedding model to determine the chunk size
    # so that chunks don't get truncated.
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer,
        separators=["\n \n", "\n\n", "\n", " ", ""],
        chunk_size=520,
        chunk_overlap=0,
    )
    loader = PyPDFLoader(pdf_file)
    docs = loader.load_and_split(text_splitter)
    return docs


def generate_embed_index(docs):
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    db = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_PERSIST_DIR,
    )
    return db

if __name__ == "__main__":
    main()