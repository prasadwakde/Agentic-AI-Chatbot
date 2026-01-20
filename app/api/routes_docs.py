from fastapi import APIRouter, UploadFile, Form, HTTPException

from app.rag.rag_service import ingest_user_document

router = APIRouter(prefix="/docs")


@router.post("/upload")
async def upload_doc(user_id: str = Form(...), file: UploadFile = None):
    if file is None:
        raise HTTPException(400, "No file uploaded")

    text = (await file.read()).decode("utf-8", errors="ignore")

    num_chunks = ingest_user_document(
        user_id=user_id,
        filename=file.filename,
        text=text
    )

    return {
        "status": "ok",
        "filename": file.filename,
        "chunks": num_chunks
    }
