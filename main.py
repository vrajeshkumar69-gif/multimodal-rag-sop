from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import fitz  # PyMuPDF
import faiss
import numpy as np
import time

# -------------------------------------------------------------------
# App initialization
# -------------------------------------------------------------------

app = FastAPI(title="Multimodal SOP RAG System")

# -------------------------------------------------------------------
# Global stores
# -------------------------------------------------------------------

# Parsed chunks from documents
DOCUMENT_STORE = []

# Embedding model
EMBEDDING_MODEL = SentenceTransformer("all-MiniLM-L6-v2")

# FAISS vector store
VECTOR_DIMENSION = 384  # all-MiniLM-L6-v2 output size
VECTOR_INDEX = faiss.IndexFlatL2(VECTOR_DIMENSION)

# Metadata aligned to FAISS vectors
VECTOR_METADATA = []

# -------------------------------------------------------------------
# Helper functions
# -------------------------------------------------------------------

def build_vector_index():
    """
    Builds or rebuilds the FAISS index from DOCUMENT_STORE
    """
    texts = []
    metadata = []

    for chunk in DOCUMENT_STORE:
        texts.append(chunk["content"])
        metadata.append({
            "type": chunk["type"],
            "page": chunk["page"],
            "source": chunk["source"]
        })

    if not texts:
        return 0

    embeddings = EMBEDDING_MODEL.encode(texts)
    embeddings = np.array(embeddings).astype("float32")

    VECTOR_INDEX.reset()
    VECTOR_INDEX.add(embeddings)

    VECTOR_METADATA.clear()
    VECTOR_METADATA.extend(metadata)

    return len(texts)

# -------------------------------------------------------------------
# API models
# -------------------------------------------------------------------

class QueryRequest(BaseModel):
    question: str

# -------------------------------------------------------------------
# Endpoints
# -------------------------------------------------------------------

@app.get("/health")
def health():
    return {
        "status": "ok",
        "documents_indexed": len(VECTOR_METADATA),
        "message": "System is up and ready"
    }


@app.post("/ingest")
async def ingest_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    start_time = time.time()

    pdf_bytes = await file.read()
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    text_chunks = 0
    table_chunks = 0
    image_chunks = 0

    for page_number, page in enumerate(doc, start=1):

        # Text
        text = page.get_text().strip()
        if text:
            DOCUMENT_STORE.append({
                "type": "text",
                "content": text,
                "page": page_number,
                "source": file.filename
            })
            text_chunks += 1

        # Images → summarized text
        images = page.get_images(full=True)
        for _ in images:
            DOCUMENT_STORE.append({
                "type": "image",
                "content": f"Image on page {page_number} showing diagrams or handling instructions.",
                "page": page_number,
                "source": file.filename
            })
            image_chunks += 1

        # Tables (best effort)
        try:
            tables = page.find_tables()
            for table in tables:
                table_text = "\n".join(
                    ["\t".join(map(str, row)) for row in table.extract()]
                )
                DOCUMENT_STORE.append({
                    "type": "table",
                    "content": table_text,
                    "page": page_number,
                    "source": file.filename
                })
                table_chunks += 1
        except Exception:
            pass

    processing_time = round(time.time() - start_time, 2)
    indexed_chunks = build_vector_index()

    return {
        "filename": file.filename,
        "text_chunks": text_chunks,
        "table_chunks": table_chunks,
        "image_chunks": image_chunks,
        "total_chunks": text_chunks + table_chunks + image_chunks,
        "indexed_chunks": indexed_chunks,
        "processing_time_seconds": processing_time
    }


@app.post("/query")
def query_document(request: QueryRequest):
    if VECTOR_INDEX.ntotal == 0:
        raise HTTPException(
            status_code=400,
            detail="No documents indexed. Please ingest a PDF first."
        )

    # Embed query
    query_embedding = EMBEDDING_MODEL.encode([request.question])
    query_embedding = np.array(query_embedding).astype("float32")

    # Retrieve top-K
    top_k = min(5, VECTOR_INDEX.ntotal)
    distances, indices = VECTOR_INDEX.search(query_embedding, top_k)

    retrieved_context = []
    retrieved_sources = []

    for idx in indices[0]:
        retrieved_context.append(DOCUMENT_STORE[idx]["content"])
        retrieved_sources.append(VECTOR_METADATA[idx])

    return {
        "question": request.question,
        "retrieved_sources": retrieved_sources,
        "context": retrieved_context
    }
