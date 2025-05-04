# services/qa_service.py
import asyncio
from concurrent.futures import ThreadPoolExecutor
import pandas as pd  # Import pandas

# --- Use your VectorStore ---
from database.vector_store import VectorStore  # Your custom class
# ---

from models import QueryRequest, QueryResponse

# --- Initialize YOUR VectorStore ---
vec = VectorStore()
# ---
thread_pool = ThreadPoolExecutor()


async def run_sync_in_thread(func, *args, **kwargs):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(thread_pool, lambda: func(*args, **kwargs))


# --- Updated to use YOUR VectorStore.search ---
def _perform_search(question: str, limit: int, source_file: str | None):
    """Performs the vector search using vec.search, returning a DataFrame."""
    try:
        metadata_filter = {}
        if source_file:
            # Key 'source_file' must match the key used during ingestion metadata storage
            metadata_filter['source_file'] = source_file

        print(
            f"Performing vector search via VectorStore for: '{question}' with filter: {metadata_filter}, limit: {limit}")

        # Call YOUR VectorStore's search method
        # It handles embedding internally and returns a DataFrame by default
        search_results_df = vec.search(
            query_text=question,
            limit=limit,
            metadata_filter=metadata_filter,
            return_dataframe=True  # Explicitly ensure DataFrame return
        )

        result_count = len(search_results_df) if isinstance(search_results_df, pd.DataFrame) else 0
        print(f"Search returned {result_count} results as DataFrame.")
        return search_results_df  # Return the DataFrame
    except Exception as e:
        print(f"Error during vector search: {e}")
        import traceback
        traceback.print_exc()
        raise ConnectionError(f"Database search failed: Check VectorStore. Error: {e}")


# --- End Update ---

# --- Updated to work with DataFrame context ---
def _simple_answer_from_context(question: str, context_df: pd.DataFrame) -> str:
    """Generates a simple answer by concatenating context from the results DataFrame."""
    if context_df is None or context_df.empty:
        return "I could not find any relevant information matching your query and filter."

    # Extract 'content' column from DataFrame
    try:
        # Ensure the column name matches the output of your VectorStore._create_dataframe_from_results
        contents = context_df['content'].astype(str).tolist()
        context_str = "\n\n---\n\n".join(contents)
        answer = f"Based on the retrieved context:\n\n{context_str}\n\nRegarding your question: '{question}'"
    except KeyError:
        print("Error: 'content' column not found in search results DataFrame.")
        answer = "Error retrieving content from search results."
    except Exception as e:
        print(f"Error formatting answer from context DataFrame: {e}")
        answer = "Error processing search results."

    return answer


# --- End Update ---

# process_query remains largely the same structure
async def process_query(request: QueryRequest) -> QueryResponse:
    """Handles the Q&A process: search, format response."""
    try:
        # 1. Perform Search (using updated _perform_search)
        search_results_df = await run_sync_in_thread(
            _perform_search,
            request.question,
            request.limit,
            request.source_file
        )

        # 2. Generate Answer (using updated context function)
        final_answer = await run_sync_in_thread(
            _simple_answer_from_context, request.question, search_results_df
        )

        # 3. Format Response
        return QueryResponse(answer=final_answer)

    except ConnectionError as db_error:
        print(f"Database error processing query: {db_error}")
        return QueryResponse(error=str(db_error))
    except Exception as e:
        print(f"An unexpected error occurred during query processing: {e}")
        import traceback
        traceback.print_exc()
        return QueryResponse(error="An unexpected internal error occurred.")
