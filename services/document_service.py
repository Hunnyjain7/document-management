# services/document_service.py
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import List

# --- Use your VectorStore ---
from database.vector_store import VectorStore  # Your custom class

# ---

# --- Initialize YOUR VectorStore ---
vec = VectorStore()
# ---
thread_pool = ThreadPoolExecutor()


async def run_sync_in_thread(func, *args, **kwargs):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(thread_pool, lambda: func(*args, **kwargs))


async def list_available_source_files() -> List[str]:
    """Retrieves a list of unique source filenames using vec.list_distinct_metadata_values."""
    print("Attempting to list distinct source files via VectorStore...")
    try:
        # IMPORTANT: Assumes you have implemented list_distinct_metadata_values in your VectorStore class
        # The field name 'source_file' MUST match what's stored in metadata
        distinct_files = await run_sync_in_thread(vec.list_distinct_metadata_values, 'source_file')
        print(f"Found {len(distinct_files)} distinct source files.")
        return distinct_files
    except AttributeError:
        print("ERROR: 'list_distinct_metadata_values' method not found in VectorStore class.")
        raise RuntimeError("Server configuration error: Cannot list documents.")
    except Exception as e:
        print(f"Error fetching distinct source files: {e}")
        import traceback
        traceback.print_exc()
        raise RuntimeError(f"Failed to retrieve document list from storage. Error: {e}")
