from fastapi import FastAPI

app = FastAPI(title="Multimodal SOP RAG System")

@app.get("/health")
def health():
    return {
        "status": "ok",
        "documents_indexed": 0,
        "message": "System is up and ready"
    }
