# models.py
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any


class IngestionResponse(BaseModel):
    message: str
    source_file: str
    chunks_added: int


class QueryRequest(BaseModel):
    question: str = Field(..., description="The user's question.")
    limit: int = Field(default=1, description="Maximum number of relevant chunks to retrieve.")
    # Simplified filter: just the source filename
    source_file: Optional[str] = Field(None,
                                       description="Filter results to only this source file (e.g., 'my_document.pdf').")


class QueryResponse(BaseModel):
    answer: Optional[str] = Field(None, description="The generated answer based on retrieved context.")
    # Optional: Include retrieved context for debugging/transparency
    # context: Optional[List[Dict[str, Any]]] = Field(None, description="Retrieved context chunks.")
    error: Optional[str] = Field(None, description="Error message if the query failed.")


class DocumentListResponse(BaseModel):
    documents: List[str] = Field(..., description="A list of unique source filenames available for querying.")
