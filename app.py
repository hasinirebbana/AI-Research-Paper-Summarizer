import streamlit as st
import PyPDF2
from docx import Document
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from collections import Counter
import fitz
from PIL import Image
import io

# ---------------- SAFE NLTK SETUP ----------------
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# ---------------- STREAMLIT UI ----------------
st.title("AI Research Paper Summarizer")

uploaded_file = st.file_uploader(
    "Upload Research Paper (PDF or DOCX)",
    type=["pdf", "docx"]
)

# ---------------- TEXT EXTRACTION ----------------
def extract_text_from_pdf(pdf_file):
    pdf_file.seek(0)
    reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

def extract_text_from_docx(docx_file):
    doc = Document(docx_file)
    text = ""
    for para in doc.paragraphs:
        text += para.text + "\n"
    return text

# ---------------- IMAGE EXTRACTION ----------------
def extract_images_from_pdf(pdf_file):
    images = []
    pdf_file.seek(0)
    pdf = fitz.open(stream=pdf_file.read(), filetype="pdf")

    for page in pdf:
        image_list = page.get_images(full=True)
        for img in image_list:
            xref = img[0]
            base_image = pdf.extract_image(xref)
            image_bytes = base_image["image"]
            image = Image.open(io.BytesIO(image_bytes))
            images.append(image)

    return images

# ---------------- SUMMARIZATION ----------------
def summarize_text(text, num_sentences=5):
    if not text.strip():
        return ""

    stop_words = set(stopwords.words("english"))
    words = word_tokenize(text.lower())
    filtered_words = [w for w in words if w.isalnum() and w not in stop_words]

    word_freq = Counter(filtered_words)
    sentences = sent_tokenize(text)

    sentence_scores = {}
    for sentence in sentences:
        for word in word_tokenize(sentence.lower()):
            if word in word_freq:
                sentence_scores[sentence] = sentence_scores.get(sentence, 0) + word_freq[word]

    sorted_sentences = sorted(sentence_scores, key=sentence_scores.get, reverse=True)
    selected_sentences = sorted_sentences[:num_sentences]

    return " ".join(selected_sentences)

# ---------------- STRUCTURED SECTION EXTRACTION ----------------
def extract_main_sections(text):
    sections = {
        "INTRODUCTION": "",
        "METHODOLOGY": "",
        "CONCLUSION": "",
        "FUTURE SCOPE": "",
        "ADDITIONAL SUMMARY": ""
    }

    current_section = "ADDITIONAL SUMMARY"

    for line in text.split("\n"):
        upper_line = line.upper()

        if "INTRODUCTION" in upper_line:
            current_section = "INTRODUCTION"
        elif "METHODOLOGY" in upper_line or "METHODS" in upper_line:
            current_section = "METHODOLOGY"
        elif "CONCLUSION" in upper_line:
            current_section = "CONCLUSION"
        elif "FUTURE SCOPE" in upper_line or "FUTURE WORK" in upper_line:
            current_section = "FUTURE SCOPE"

        sections[current_section] += line + " "

    return sections

# ---------------- IMAGE CONTEXT EXTRACTION ----------------
def extract_image_related_text(text):
    lines = text.split("\n")
    image_related_text = ""
    capture_next = 0

    for line in lines:
        if "FIG." in line.upper() or "FIGURE" in line.upper():
            image_related_text += line + " "
            capture_next = 4

        elif capture_next > 0:
            image_related_text += line + " "
            capture_next -= 1

    return image_related_text

# ---------------- MAIN ----------------
if uploaded_file:

    extracted_text = ""
    images = []

    if uploaded_file.name.endswith(".pdf"):
        extracted_text = extract_text_from_pdf(uploaded_file)
        images = extract_images_from_pdf(uploaded_file)
    else:
        extracted_text = extract_text_from_docx(uploaded_file)

    # -------- SHOW EXTRACTED TEXT --------
    st.subheader("Extracted Text")
    st.text_area("Raw Extracted Content", extracted_text, height=250)

    if st.button("Generate Structured Summary"):
        with st.spinner("Generating summary..."):
            st.subheader("Overall Summary")
            st.write(summarize_text(extracted_text, 8))

            sections = extract_main_sections(extracted_text)

        # -------- INTRODUCTION --------
            st.subheader("Introduction")
            st.write(summarize_text(sections["INTRODUCTION"], 5))

        # -------- METHODOLOGY --------
            st.subheader("Methodology")
            st.write(summarize_text(sections["METHODOLOGY"], 5))

        # -------- CONCLUSION --------
            st.subheader("Conclusion")
            st.write(summarize_text(sections["CONCLUSION"], 4))

        # -------- FUTURE SCOPE --------
            st.subheader("Future Scope")
            st.write(summarize_text(sections["FUTURE SCOPE"], 4))
        # -------- ADDITIONAL SUMMARY --------
            st.subheader("Additional Summary")
            additional_summary = summarize_text(sections["ADDITIONAL SUMMARY"], 6)
            for sentence in sent_tokenize(additional_summary):
                st.write("• " + sentence)

        # -------- IMAGE SUMMARY --------
        if images:
            st.subheader("Image Summary")

            image_text = extract_image_related_text(extracted_text)

            if image_text.strip():
                st.write(summarize_text(image_text, 4))
            else:
                st.write("Image-related descriptions are limited in the text.")

            # -------- SHOW IMAGES SIDE BY SIDE --------
            st.subheader("Extracted Images")

            num_cols = 3  # 3 images per row
            cols = st.columns(num_cols)

            for idx, img in enumerate(images):
                with cols[idx % num_cols]:
                    st.image(img, caption=f"Image {idx+1}", width=220)
