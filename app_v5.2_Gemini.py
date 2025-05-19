# VERSION_5.2_Gemini- ATTEMPT TO generate Document with user providing the template file for the document
# The content generated will follow the template provided by the user

# This helps user to get the desired output by his desired template
# Also the previous models were very slow in processing because of Ollama ,overcomed by using Gemini API and HuggingFace 


#Knowledge Base- Company support pdf Resource ,related company webPage(Web Scrapping included)

import streamlit as st
import os
import requests
from bs4 import BeautifulSoup
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from reportlab.lib.pagesizes import LETTER
from reportlab.pdfgen import canvas
from textwrap import wrap
from datetime import datetime
from jinja2 import Template
from langchain.schema import Document
import google.generativeai as genai
import torch

# --- Constants ---
PREDEFINED_PDF_FILES = ["dog_allergies.pdf", "unfiltered_dogs.pdf"]
PREDEFINED_WEB_URLS = [
    "https://www.hsallergy.com/allergy-extracts/Dog-Hair-Extracts/"
]
GEMINI_API_KEY = "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"  


genai.configure(api_key=GEMINI_API_KEY)

# --- Generate PDF Output ---
def generate_pdf(text, filename="output.pdf"):
    c = canvas.Canvas(filename, pagesize=LETTER)
    width, height = LETTER
    textobject = c.beginText(40, height - 50)
    textobject.setFont("Helvetica", 12)
    for line in text.split('\n'):
        wrapped = wrap(line, width=90)
        for w in wrapped:
            textobject.textLine(w)
    c.drawText(textobject)
    c.save()

# --- Load PDFs ---
def load_pdfs(filepaths):
    docs = []
    for filepath in filepaths:
        loader = PyPDFLoader(filepath)
        docs.extend(loader.load())
    return docs

# --- Scrape Webpage ---
def scrape_webpage(url):
    response = requests.get(url, timeout=10)
    soup = BeautifulSoup(response.content, "html.parser")
    text = soup.get_text(separator="\n")
    return text

# --- Create Combined Vectorstore ---
def create_combined_vectorstore(pdf_docs, web_texts):
    all_documents = []
    for doc in pdf_docs:
        if isinstance(doc, dict):
            all_documents.append(Document(page_content=doc.get('page_content', ''), metadata=doc.get('metadata', {})))
        else:
            all_documents.append(doc)

    for web_text in web_texts:
        all_documents.append(Document(page_content=web_text, metadata={"source": "web"}))

    
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(all_documents)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": device}
    )

    vectorstore = Chroma.from_documents(chunks, embedding=embeddings)
    return vectorstore.as_retriever()


def search_relevant_context(retriever, query):
    results = retriever.get_relevant_documents(query)
    combined_context = "\n\n".join([doc.page_content for doc in results])
    return combined_context


def generate_text_with_gemini(prompt):
    model = genai.GenerativeModel(model_name="models/gemini-1.5-pro-latest")
    response = model.generate_content(prompt)
    return response.text

# --- Generate Response from Template ---
def generate_response_from_template(template_str, retriever, user_input, meta={}):
    
    context = search_relevant_context(retriever, user_input)
    meta["context"] = context

    
    template = Template(template_str)
    filled_prompt = template.render(**meta)

 
    generated_text = generate_text_with_gemini(filled_prompt)
    return generated_text


def collect_feedback():
    st.subheader("Please rate the document")
    quality = st.slider("Quality (1 = Poor, 5 = Excellent)", 1, 5, 3)
    coherence = st.slider("Coherence (1 = Poor, 5 = Excellent)", 1, 5, 3)
    relevance = st.slider("Relevance (1 = Poor, 5 = Excellent)", 1, 5, 3)
    grammar = st.slider("Grammatical Correctness (1 = Poor, 5 = Excellent)", 1, 5, 3)
    satisfaction = st.slider("Overall Satisfaction (1 = Poor, 5 = Excellent)", 1, 5, 3)

    feedback = {
        "quality": quality,
        "coherence": coherence,
        "relevance": relevance,
        "grammar": grammar,
        "satisfaction": satisfaction
    }

    if st.button("Submit Feedback"):
        st.success("Thank you for your feedback!")
        print(feedback)

# --- Streamlit UI ---
st.set_page_config(page_title="GenAI Document Generation Bot", layout="centered")
st.title("GenAI Document Generation Bot")


with st.spinner("Building knowledge base..."):
    pdf_docs = load_pdfs(PREDEFINED_PDF_FILES)
    web_texts = [scrape_webpage(url) for url in PREDEFINED_WEB_URLS]
    retriever = create_combined_vectorstore(pdf_docs, web_texts)

# --- Upload Template ---
template_file = st.file_uploader("Upload a Template File (.txt or .md)", type=["txt", "md"])

if template_file:
    template_str = template_file.read().decode("utf-8")

    query = st.text_area("Enter your query for the document")

    meta_inputs = {
        "date": datetime.now().strftime("%B %d, %Y")
    }

    st.markdown("**Optional Template Fields (Other than context or date):**")
    field_names = st.text_input("Enter custom fields (comma-separated)", value="title,recipient,topic")
    for field in [f.strip() for f in field_names.split(",") if f.strip() and f not in ["context", "date"]]:
        meta_inputs[field] = st.text_input(f"Value for '{field}'")

    if st.button("Generate Document"):
        with st.spinner("Generating document..."):
            try:
                response = generate_response_from_template(template_str, retriever, query, meta_inputs)
                st.subheader("Generated Output")
                st.code(response, language="markdown")

                filename = f"generated_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
                generate_pdf(response, filename)

                with open(filename, "rb") as pdf_file:
                    st.download_button("Download PDF", pdf_file, file_name=filename)

            except Exception as e:
                st.error(str(e))

        collect_feedback()
