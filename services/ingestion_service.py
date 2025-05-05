import os
from datetime import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
from fastapi import UploadFile, HTTPException, status
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from database.vector_store import VectorStore  # Your custom class
from timescale_vector.client import uuid_from_time  # Your specific library

from config.settings import TEMP_FOLDER, MAX_FILE_SIZE_BYTES, CHUNK_SIZE, CHUNK_OVERLAP

# --- Initialize YOUR VectorStore ---
vec = VectorStore()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    length_function=len,
    is_separator_regex=False,
)
thread_pool = ThreadPoolExecutor()


async def run_sync_in_thread(func, *args, **kwargs):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(thread_pool, lambda: func(*args, **kwargs))


async def save_temp_file_with_limit(file: UploadFile) -> tuple[str, str]:
    if not file.filename:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No filename provided.")
    original_filename = file.filename
    temp_filename = f"temp_{datetime.now().timestamp()}_{original_filename}"
    file_path = os.path.join(TEMP_FOLDER, temp_filename)
    try:
        size = 0
        with open(file_path, "wb") as buffer:
            while chunk := await file.read(8192):
                size += len(chunk)
                if size > MAX_FILE_SIZE_BYTES:
                    os.remove(file_path)
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail=f"File size {size / (1024 * 1024):.2f} MB exceeds limit of {MAX_FILE_SIZE_BYTES / (1024 * 1024):.2f} MB."
                    )
                buffer.write(chunk)
        print(f"File '{original_filename}' temporarily saved to: {file_path}, Size: {size} bytes")
        return file_path, original_filename
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        print(f"Error saving file: {e}")
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Could not save file: {e}")
    finally:
        await file.close()


def _load_and_split(file_path: str, source_filename: str):
    print(f"Loading and splitting: {file_path}")
    try:
        loader = UnstructuredFileLoader(file_path)
        docs = loader.load()
        if not docs: return []
        for doc in docs:
            doc.metadata['source_file'] = source_filename
            doc.metadata['created_at'] = datetime.now().isoformat()
        chunks = text_splitter.split_documents(docs)
        print(f"Split '{source_filename}' into {len(chunks)} chunks.")
        return chunks
    except Exception as e:
        print(f"Error loading/splitting file {file_path}: {e}")
        raise ValueError(f"Failed to process file content: {e}")


def _prepare_and_upsert_chunks(chunks):
    """Generates embeddings using vec.get_embedding, formats as DataFrame, and upserts via vec.upsert."""
    if not chunks:
        return 0

    records_list = []
    print(f"Preparing {len(chunks)} chunks for database...")
    for i, chunk in enumerate(chunks):  # Keep 'i' for logging if needed
        try:
            embedding = vec.get_embedding(chunk.page_content)

            # --- MODIFIED LINE ---
            # Generate a standard UUIDv1 using the library function
            record_id = str(uuid_from_time(datetime.now()))
            # --- END MODIFICATION ---

            record = {
                "id": record_id,  # Use the valid UUID string
                "contents": chunk.page_content,
                "embedding": embedding,
                "metadata": chunk.metadata
            }
            records_list.append(record)
        except Exception as e:
            print(f"Error processing chunk {i} ('{chunk.metadata.get('source_file', 'unknown')}'): {e}")
            continue

    if not records_list:
        print("No chunks were successfully prepared for upsert.")
        return 0

    try:
        records_df = pd.DataFrame(records_list)
        print(f"Upserting {len(records_df)} processed chunks via DataFrame...")
        vec.upsert(records_df)  # Pass the DataFrame with valid UUIDs
        print("Upsert successful.")
        return len(records_df)
    except Exception as e:
        print(f"Error during database upsert: {e}")
        import traceback
        traceback.print_exc()
        raise ConnectionError(f"Database upsert failed: Check VectorStore logs and DB connection. Error: {e}")


async def ingest_document(file: UploadFile):
    file_path = None
    original_filename = "unknown"
    try:
        file_path, original_filename = await save_temp_file_with_limit(file)
        chunks = await run_sync_in_thread(_load_and_split, file_path, original_filename)
        chunks_added_count = await run_sync_in_thread(_prepare_and_upsert_chunks, chunks)
        return {
            "message": f"Document '{original_filename}' processed.",
            "source_file": original_filename,
            "chunks_added": chunks_added_count
        }
    except (ValueError, ConnectionError) as processing_error:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(processing_error))
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        print(f"An unexpected error occurred during ingestion of '{original_filename}': {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=f"Internal server error during ingestion: {e}")
    finally:
        if file_path and os.path.exists(file_path):
            try:
                await run_sync_in_thread(os.remove, file_path)
                print(f"Temporary file removed: {file_path}")
            except OSError as e:
                print(f"Error removing temporary file {file_path}: {e}")
