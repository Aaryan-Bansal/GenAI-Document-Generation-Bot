# VERSION_6- ATTEMPT TO generate Document with giving the user the option to
# Select the DOC. type he needs, and get the output using the corresponding template file stored (Not hardcoded template)

# It helps the user to select the doc. type without needing to upload a template by himself


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
TEMPLATES = {
    "Report": "templates/report_template.txt", 
    "Letter": "templates/letter_template.txt",
    "Summary": "templates/summary_template.txt"
}

PREDEFINED_PDF_FILES = ["dog_allergies.pdf", "unfiltered_dogs.pdf"]
PREDEFINED_WEB_URLS = [
    "https://www.hsallergy.com/allergy-extracts/Dog-Hair-Extracts/"
]
GEMINI_API_KEY = "AIzaSyBnPMtmYL2H-6A3ujlW6A-UKdV9348nTVc"  

genai.configure(api_key=GEMINI_API_KEY)


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


def load_pdfs(filepaths):
    docs = []
    for filepath in filepaths:
        loader = PyPDFLoader(filepath)
        docs.extend(loader.load())
    return docs


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


def generate_response_from_template(template_str, retriever, query, meta_inputs):
    context = search_relevant_context(retriever, query)
    meta_inputs["context"] = context

    template = Template(template_str)
    filled_prompt = template.render(**meta_inputs)

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


doc_type = st.selectbox("Select Document Type", options=list(TEMPLATES.keys()))
selected_template_file = TEMPLATES[doc_type]


with open(selected_template_file, "r") as file:
    template_str = file.read()

st.subheader(f"Preview of {doc_type} Template")
st.code(template_str, language="markdown")


query = st.text_area("Enter document query")

meta_inputs = {
    "date": datetime.now().strftime("%B %d, %Y")
}


if doc_type == "Report":
    meta_inputs["title"] = st.text_input("Report Title")
    meta_inputs["author"] = st.text_input("Author Name")
elif doc_type == "Letter":
    meta_inputs["recipient"] = st.text_input("Recipient Name")
    meta_inputs["subject"] = st.text_input("Letter Subject")
elif doc_type == "Summary":
    meta_inputs["summary_title"] = st.text_input("Summary Title")

if st.button("Generate Document"):
    try:
        response = generate_response_from_template(template_str, retriever, query, meta_inputs)
        st.subheader(f"Generated {doc_type}")
        st.code(response, language="markdown")

        filename = f"generated_{doc_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        generate_pdf(response, filename)


        with open(filename, "rb") as pdf_file:
            st.download_button("Download PDF", pdf_file, file_name=filename)

    except Exception as e:
        st.error(f"Error: {str(e)}")

collect_feedback()
