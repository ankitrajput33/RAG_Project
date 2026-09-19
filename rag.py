from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# 1. Read the document
with open("documents/new_document.txt", "r", encoding="utf-8") as file:
    text = file.read()


# 2. Split document into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = text_splitter.split_text(text)


# 3. Create embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# 4. Create vector database
vectorstore = Chroma.from_texts(
    texts=chunks,
    embedding=embeddings,
    persist_directory="chroma_db"
)


# 5. Load the LLM
tokenizer = AutoTokenizer.from_pretrained(
    "google/flan-t5-small"
)

model = AutoModelForSeq2SeqLM.from_pretrained(
    "google/flan-t5-small"
)


# 6. Chatbot loop
while True:

    query = input("\nAsk a question (type 'exit' to quit): ")

    if query.lower() == "exit":
        print("Chatbot closed.")
        break


    # 7. Retrieve relevant information
    results = vectorstore.similarity_search(query, k=1)

    context = results[0].page_content

    print("\nRetrieved Context:")
    print(context)


    # 8. Create prompt
    prompt = f"""
Answer the question using only the information in the context.

Context:
{context}

Question:
{query}

Answer:
"""


    # 9. Generate answer
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


    # 10. Display answer
    print("\nFinal Answer:")
    print(answer)