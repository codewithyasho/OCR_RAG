from pdf2image import convert_from_bytes
import streamlit as st
from langchain_ollama import ChatOllama
from pathlib import Path
import pytesseract
import cv2

# Tesseract path
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

st.title("OCR RAG Document QA")


uploaded_file = st.file_uploader("Upload PDF", type="pdf")

if uploaded_file is not None:

    # STEP 1: Convert PDF to images
    images = convert_from_bytes(
        uploaded_file.read(),
        dpi=300,
        poppler_path=r"C:\Users\yasho\Downloads\Release-25.12.0-0\poppler-25.12.0\Library\bin"
    )

    data_folder = Path("data")
    data_folder.mkdir(exist_ok=True)

    for i, img in enumerate(images):
        img_path = data_folder / f"page_{i+1}.png"
        img.save(img_path, "PNG")

    st.success(f"{len(images)} images saved in 'data' folder")

    # STEP 2: OCR
    all_text = ""

    for img_file in sorted(data_folder.glob("*.png")):

        img = cv2.imread(str(img_file))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        text = pytesseract.image_to_string(gray)

        all_text += text + "\n"

    st.success("Text extracted from images")

    # STEP 3: Ask question
    llm = ChatOllama(model="qwen3-vl:235b-cloud")

    question = st.text_input("Ask question about the document")

    if question:

        prompt = f"""
        Answer the question based on the document below ONLY.

        Document:
        {all_text}

        Question:
        {question}
        """

        response = llm.invoke(prompt)

        st.write(response.content)
