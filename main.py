from fastapi import FastAPI

app = FastAPI(title="Multimodal SOP RAG System")

@app.get("/health")
def health():
    return {
        "status": "ok",
        "documents_indexed": 0,
        "message": "System is up and ready"
    }

from fastapi import UploadFile, File, HTTPException
import fitz  # PyMuPDF
import time

# In-memory store for parsed documents
DOCUMENT_STORE = []


@app.post("/ingest")
async def ingest_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    start_time = time.time()

    try:
        pdf_bytes = await file.read()
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read PDF: {str(e)}")

    text_chunks = 0
    table_chunks = 0
    image_chunks = 0

    for page_number, page in enumerate(doc, start=1):

        # Extract text
        text = page.get_text().strip()
        if text:
            DOCUMENT_STORE.append({
                "type": "text",
                "content": text,
                "page": page_number,
                "source": file.filename
            })
            text_chunks += 1

        # Extract images (represented as summaries for now)
        images = page.get_images(full=True)
        for _ in images:
            DOCUMENT_STORE.append({
                "type": "image",
                "content": f"Image on page {page_number} showing diagrams or handling instructions.",
                "page": page_number,
                "source": file.filename
            })
            image_chunks += 1

        # Extract tables (best-effort, treated as text)
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

    return {
        "filename": file.filename,
        "text_chunks": text_chunks,
        "table_chunks": table_chunks,
        "image_chunks": image_chunks,
        "total_chunks": text_chunks + table_chunks + image_chunks,
        "processing_time_seconds": processing_time
    }
