from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from pdf2image import convert_from_bytes
from langchain_ollama import ChatOllama
from pathlib import Path
import streamlit as st
import pytesseract
import shutil
import cv2


# paths
POPPLER_PATH = r"C:\Users\yasho\Downloads\Release-25.12.0-0\poppler-25.12.0\Library\bin"
TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

st.title("OCR RAG Document QA")

uploaded_file = st.file_uploader("Upload PDF", type="pdf")


if uploaded_file:

    # clear old data
    data_folder = Path("data")

    if data_folder.exists():
        shutil.rmtree(data_folder)

    data_folder.mkdir()

    # ======================
    # STEP 1 PDF → IMAGES
    # ======================

    images = convert_from_bytes(
        uploaded_file.read(),
        dpi=300,
        poppler_path=POPPLER_PATH
    )

    for i, img in enumerate(images):
        img.save(data_folder / f"page_{i+1}.png")

    st.success("PDF converted to images")

    # ======================
    # STEP 2 OCR
    # ======================

    all_text = ""

    for img_file in sorted(data_folder.glob("*.png")):

        img = cv2.imread(str(img_file))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        text = pytesseract.image_to_string(gray)

        all_text += text + "\n"

    st.success("OCR completed")

    # ======================
    # STEP 3 CHUNKING
    # ======================

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )

    chunks = splitter.split_text(all_text)

    st.write("Chunks created:", len(chunks))

    # ======================
    # STEP 4 EMBEDDINGS
    # ======================

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # ======================
    # STEP 5 VECTOR DB
    # ======================

    vector_db = Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
        persist_directory="vector_db"
    )

    st.success("Vector DB created")

    # ======================
    # STEP 6 QUESTION
    # ======================

    question = st.text_input("Ask a question about the document")

    if question:

        docs = vector_db.similarity_search(question, k=3)

        context = "\n".join([doc.page_content for doc in docs])

        llm = ChatOllama(model="qwen3:latest")

        prompt = f"""
        Answer the question using the context below.

        Context:
        {context}

        Question:
        {question}
        """

        response = llm.invoke(prompt)

        st.write(response.content)
