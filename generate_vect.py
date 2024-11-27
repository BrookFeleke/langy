import os
import random
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_experimental.text_splitter import SemanticChunker 
from pydantic import BaseModel
from unstructured.partition.pdf import partition_pdf
from pytesseract import pytesseract
from typing import Any

docs = []
load_dotenv()
# Access the API key from environment variables
TESSERACT_PATH = os.getenv('TESSERACT_PATH')
pytesseract.tesseract_cmd = f'{TESSERACT_PATH}'
def parse_pdf(filePath:str):
    raw_pdf_elements = partition_pdf(
    filename=filePath,
    extract_images_in_pdf=False,
    infer_table_structure=True,
    chunking_strategy="by_title",
    max_characters=4000,
    new_after_n_chars=3800,
    combine_text_under_n_chars=2000,
    image_output_dir_path='.',
)
    return raw_pdf_elements
def get_Docs():
    raw_pdf_elements = parse_pdf('./pdfs/CPE.pdf')
    class Element(BaseModel):
        type: str
        text: Any
    print(raw_pdf_elements)
# Categorize by type
    categorized_elements = []
    for element in raw_pdf_elements:
        if "unstructured.documents.elements.Table" in str(type(element)):
            categorized_elements.append(Element(type="table", text=str(element)))
        elif "unstructured.documents.elements.CompositeElement" in str(type(element)):
            categorized_elements.append(Element(type="text", text=str(element)))
# Text
    text_elements = [e for e in categorized_elements if e.type == "text"]
    print(len(text_elements))
    return text_elements
def run ():
    result = get_Docs()
    print(type(result[0]))
    print(type(result))
    embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'})
    text_splitter = SemanticChunker(embedding_function,breakpoint_threshold_type="percentile")
    docs = [e.text for e in result] 
    docs = [text_splitter.split_text(text) for text in docs]
    texts = []
    for doc in docs:
        for text in doc:
            texts.append(text)

    vectorstore = Chroma.from_texts(texts, embedding_function, persist_directory="./comprog_chroma_db")
    return vectorstore
    
run()
