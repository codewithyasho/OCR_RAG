from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from pdf2image import convert_from_bytes
from langchain_ollama import ChatOllama
from pathlib import Path
import streamlit as st
import pytesseract
import shutil
import cv2

POPPLER_PATH = r"C:\Users\yasho\Downloads\Release-25.12.0-0\poppler-25.12.0\Library\bin"
TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

st.title("OCR RAG Document QA")

uploaded_file = st.file_uploader("Upload PDF", type="pdf")


@st.cache_resource
def build_vector_db(all_text):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )

    chunks = splitter.split_text(all_text)

    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en"
    )

    vector_db = Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
        persist_directory="vector_db"
    )

    return vector_db


if uploaded_file:

    data_folder = Path("data")

    if data_folder.exists():
        shutil.rmtree(data_folder)

    data_folder.mkdir()

    # STEP 1 PDF → Images
    images = convert_from_bytes(
        uploaded_file.read(),
        dpi=300,
        poppler_path=POPPLER_PATH
    )

    for i, img in enumerate(images):
        img.save(data_folder / f"page_{i+1}.png")

    st.success("PDF converted to images")

    # STEP 2 OCR
    all_text = ""

    for img_file in sorted(data_folder.glob("*.png")):

        img = cv2.imread(str(img_file))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        text = pytesseract.image_to_string(gray)

        all_text += text + "\n"

    st.success("OCR completed")

    # STEP 3 Build Vector DB (cached)
    vector_db = build_vector_db(all_text)

    st.success("Vector DB ready")

    question = st.text_input("Ask a question about the document")

    if question:

        docs = vector_db.similarity_search(question, k=5)

        context = "\n".join([doc.page_content for doc in docs])

        llm = ChatOllama(model="deepseek-v3.1:671b-cloud")

        prompt = f"""
        Answer using ONLY the context below.
        If the answer is not in the context, say "I don't know".

        Context:
        {context}

        Question:
        {question}
        """

        response = llm.invoke(prompt)

        st.write(response.content)
