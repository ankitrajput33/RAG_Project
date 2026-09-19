import streamlit as st

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# Page settings
st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖"
)

st.title("🤖 RAG Chatbot")
st.write("Ask questions about your document.")


# Load document
with open("documents/new_document.txt", "r", encoding="utf-8") as file:
    text = file.read()


# Split document
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = text_splitter.split_text(text)


# Create embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# Create vector database
vectorstore = Chroma.from_texts(
    texts=chunks,
    embedding=embeddings,
    persist_directory="chroma_db"
)


# Load LLM
tokenizer = AutoTokenizer.from_pretrained(
    "google/flan-t5-small"
)

model = AutoModelForSeq2SeqLM.from_pretrained(
    "google/flan-t5-small"
)


# Chat input
question = st.chat_input("Ask a question...")


if question:

    # Retrieve relevant context
    results = vectorstore.similarity_search(
        question,
        k=1
    )

    context = results[0].page_content


    # Create prompt
    prompt = f"""
Answer the question using only the information in the context.

Context:
{context}

Question:
{question}

Answer:
"""


    # Generate answer
    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    )

    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        num_beams=4
    )

    answer = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    ).strip()


    # Display chat
    st.chat_message("user").write(question)

    st.chat_message("assistant").write(answer)