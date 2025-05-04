import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException, status, Body

from config.settings import MAX_FILE_SIZE_MB, TEMP_FOLDER
from services import ingestion_service, qa_service, document_service
from models import IngestionResponse, QueryRequest, QueryResponse, DocumentListResponse

app = FastAPI(
    title="Document RAG API (Text Files)",
    description=(
        "API for ingesting text documents, chunking, storing with metadata (source_file, created_at), and performing "
        "RAG Q&A with source file filtering."
    ),
    version="1.2.0",
)


@app.on_event("startup")
async def startup_event():
    print("Application startup...")
    print(f"Temporary folder: {TEMP_FOLDER}")


@app.post("/ingest/",
          summary="Ingest a Text Document",
          description=f"""Upload a text-based document (e.g., .txt, .pdf, .md, .docx).
          The document will be chunked, embedded, and stored with 'source_file' and 'created_at' metadata.
          Maximum file size: {MAX_FILE_SIZE_MB} MB.""",
          response_model=IngestionResponse,
          status_code=status.HTTP_201_CREATED,
          tags=["Ingestion"])
async def ingest_document_endpoint(file: UploadFile = File(..., description="The document file to ingest.")):
    """
    Handles document ingestion, chunking, and storage.
    """
    try:
        result = await ingestion_service.ingest_document(file)
        if result["chunks_added"] == 0 and "processed" in result["message"].lower():
            # Indicate if file was processed but resulted in no chunks (e.g., empty file)
            return IngestionResponse(**result)  # Return 201 but with 0 chunks added
        elif result["chunks_added"] == 0:
            # Should likely be caught as an error during processing if file wasn't empty
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                                detail="File processed but no content chunks were generated or stored.")

        return IngestionResponse(**result)
    except HTTPException as e:
        raise e  # Re-raise validation, size limit, or processing errors
    except Exception as e:
        print(f"Unexpected error during ingestion endpoint: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail="An unexpected error occurred during file ingestion.")


@app.get("/documents/",
         summary="List Ingested Source Files",
         description="Retrieves a list of unique source filenames from ingested documents, usable for filtering in the '/query/' endpoint.",
         response_model=DocumentListResponse,
         tags=["Documents"])
async def list_documents_endpoint():
    """
    Provides a list of unique 'source_file' values present in the vector store.
    """
    try:
        document_list = await document_service.list_available_source_files()
        return DocumentListResponse(documents=document_list)
    except RuntimeError as e:
        print(f"Error getting document list: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
    except Exception as e:
        print(f"Unexpected error listing documents: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail="An unexpected error occurred while retrieving the document list.")


@app.post("/query/",
          summary="Ask a Question (RAG)",
          description="Ask a question based on ingested documents. Optionally filter by 'source_file'.",
          response_model=QueryResponse,
          tags=["Q&A"])
async def query_endpoint(request: QueryRequest = Body(...)):
    """
    Handles user questions using RAG. Filter results using the `source_file` field.
    """
    if not request.question:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Question cannot be empty.")

    response = await qa_service.process_query(request)

    if response.error:
        status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        if "Database" in response.error or "Connection" in response.error:
            status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        raise HTTPException(status_code=status_code, detail=response.error)

    return response


@app.get("/", summary="Health Check", tags=["General"])
async def read_root():
    return {"status": "ok", "message": "Welcome to the Document RAG API!"}


# Run: uvicorn main:app --reload --port 8000
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
