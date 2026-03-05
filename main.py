from pdf2image import convert_from_bytes
import streamlit as st
import os
from pathlib import Path

uploaded_file = st.file_uploader("Upload PDF", type="pdf")

if uploaded_file is not None:

    images = convert_from_bytes(
        uploaded_file.read(),
        dpi=300,
        poppler_path=r"C:\Users\yasho\Downloads\Release-25.12.0-0\poppler-25.12.0\Library\bin"
    )

    # create data folder
    data_folder = Path("data")
    data_folder.mkdir(exist_ok=True)

    # save images
    for i, img in enumerate(images):
        img_path = data_folder / f"page_{i+1}.png"
        img.save(img_path, "PNG")

    st.success(f"{len(images)} images saved successfully in 'data' folder.")
