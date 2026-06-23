# app.py
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import streamlit as st
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM
)
from PyPDF2 import PdfReader


@st.cache_resource
def load_embedding_model():

    return SentenceTransformer(
        "all-MiniLM-L6-v2"
    )


def create_chunks(
    text,
    chunk_size=500,
    overlap=100
):

    chunks = []

    start = 0

    while start < len(text):

        chunks.append(
            text[start:start + chunk_size]
        )

        start += (
            chunk_size - overlap
        )

    return chunks


st.title("PDF RAG Chatbot")


@st.cache_resource
def load_llm():

    tokenizer = AutoTokenizer.from_pretrained(
        "google/flan-t5-large"
    )

    model = AutoModelForSeq2SeqLM.from_pretrained(
        "google/flan-t5-large"
    )

    return tokenizer, model


embedding_model = load_embedding_model()
tokenizer, llm_model = load_llm()


def generate_answer(
    question,
    context,
    tokenizer,
    model
):

    prompt = f"""
    You are a question-answering expert assistant.

    Answer ONLY using the provided context.

    If the answer is not present in the context, say: "Answer not found in the document."

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=2000
    )

    outputs = model.generate(
        **inputs,
        max_new_tokens=150
    )

    answer = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    return answer


uploaded_pdf = st.file_uploader(
    "Upload PDF",
    type=["pdf"]
)
if uploaded_pdf:

    reader = PdfReader(uploaded_pdf)

    text = ""

    for page in reader.pages:

        page_text = page.extract_text()

        if page_text:
            text += page_text

    st.success("PDF Loaded")

    st.write(
        f"Characters extracted: {len(text)}"
    )

    chunks = create_chunks(
        text,
        chunk_size=1000,
        overlap=300
    )

    embeddings = embedding_model.encode(chunks, convert_to_numpy=True)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(
        dimension
    )
    index.add(
        embeddings.astype("float32")
    )
    st.write(
        f"Embedding Shape: {embeddings.shape}"
    )
    st.write(
        f"Chunks Generated: {len(chunks)}"
    )

    st.write(
        "Chunk Size Used: 500"
    )

    question = st.text_input(
        "Ask a question about the PDF"
    )
    if question:
        query_embedding = embedding_model.encode(
            [question],
            convert_to_numpy=True
        )

        distances, indices = index.search(
            query_embedding,
            5
        )

        retrieved_chunks = [
            chunks[i]
            for i in indices[0]
        ]

        context = "\n".join(retrieved_chunks)

        st.subheader(
            "Context Sent To LLM"
        )

        st.text_area(
            "",
            context,
            height=300
        )

        answer = generate_answer(
            question,
            context,
            tokenizer,
            llm_model
        )
        st.subheader("Answer")

        st.write(answer)

        for rank, idx in enumerate(indices[0]):

            st.write(
                f"### Chunk {rank + 1}"
            )

            st.write(
                f"Distance: {distances[0][rank]:.4f}"
            )

            st.write(
                chunks[idx]
            )

            st.divider()
    st.text_area(
        "Preview",
        text[:3000],
        height=300
    )

    st.text_area(
        "First Chunk",
        chunks[0],
        height=250
    )
