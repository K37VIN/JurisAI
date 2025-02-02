import os
import json
import pdfplumber
import time
import concurrent.futures
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Paths
VECTOR_STORE_PATH = "my_vector_store"
METADATA_FILE = os.path.join(VECTOR_STORE_PATH, "metadata.json")
PDF_DIRECTORY = "LEGAL-DATA"

# Initialize embedding model once
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def extract_text_chunks(pdf_path, chunk_size=1000, chunk_overlap=200):
    text_chunks = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            text = ""
            for page in pdf.pages:
                extracted_text = page.extract_text()
                if extracted_text:
                    text += extracted_text + "\n"

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            text_chunks = text_splitter.split_text(text)

    except Exception as e:
        print(f"❌ Error processing {pdf_path}: {e}")

    return text_chunks

def process_single_pdf(filepath):
    chunks = extract_text_chunks(filepath)
    return [Document(page_content=chunk, metadata={"source": os.path.basename(filepath)}) for chunk in chunks]

def get_existing_files():
    if os.path.exists(METADATA_FILE):
        with open(METADATA_FILE, "r") as f:
            return json.load(f)
    return {}

def save_file_metadata(file_list):
    os.makedirs(VECTOR_STORE_PATH, exist_ok=True)
    with open(METADATA_FILE, "w") as f:
        json.dump(file_list, f)

def update_faiss_if_new_files():
    start_time = time.time()

    existing_files = get_existing_files()
    current_files = {file: os.path.getmtime(os.path.join(PDF_DIRECTORY, file)) 
                    for file in os.listdir(PDF_DIRECTORY) 
                    if file.endswith(".pdf")}

    new_files = [file for file in current_files 
                 if file not in existing_files or current_files[file] > existing_files[file]]

    if not new_files:
        print("✅ No new documents found. FAISS is up to date.")
        return

    print(f"🔄 Updating FAISS with {len(new_files)} new/updated files...")

    new_docs = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        pdf_paths = [os.path.join(PDF_DIRECTORY, file) for file in new_files]
        results = executor.map(process_single_pdf, pdf_paths)
        for docs in results:
            new_docs.extend(docs)

    if not new_docs:
        print("❌ No valid text extracted. Skipping FAISS update.")
        return

    texts = [doc.page_content for doc in new_docs]
    metadatas = [doc.metadata for doc in new_docs]
    embeddings = embedding_model.encode(texts, convert_to_numpy=True).tolist()

    if os.path.exists(os.path.join(VECTOR_STORE_PATH, "index.faiss")):
        db = FAISS.load_local(VECTOR_STORE_PATH, embedding_model, allow_dangerous_deserialization=True)
        # Create a list of (text, embedding) tuples
        text_embeddings = list(zip(texts, embeddings))
        db.add_embeddings(text_embeddings, metadatas)
    else:
        # Create a list of (text, embedding) tuples
        text_embeddings = list(zip(texts, embeddings))
        db = FAISS.from_embeddings(text_embeddings, metadatas)

    db.save_local(VECTOR_STORE_PATH)
    save_file_metadata(current_files)

    end_time = time.time()
    print(f"✅ FAISS updated successfully in {round(end_time - start_time, 2)} seconds.")

if __name__ == "__main__":
    update_faiss_if_new_files()