import asyncio
import logging

import ollama
from concurrent.futures import ThreadPoolExecutor

import openai
import pandas as pd

from config.settings import OLLAMA_CHAT_MODEL, OPENAI_CHAT_MODEL, OPENAI_API_KEY
from database.vector_store import VectorStore  # Your custom class

from models import QueryRequest, QueryResponse

# --- Initialize OpenAI Client ---
# Handle missing API key gracefully
if OPENAI_API_KEY:
    openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
    logging.info(f"OpenAI client initialized for model: {OPENAI_CHAT_MODEL}")
else:
    openai_client = None
    logging.warning("OpenAI client NOT initialized due to missing API key.")

# --- Initialize YOUR VectorStore ---
vec = VectorStore()
thread_pool = ThreadPoolExecutor()


async def run_sync_in_thread(func, *args, **kwargs):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(thread_pool, lambda: func(*args, **kwargs))


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


def _generate_ollama_answer(question: str, context_df: pd.DataFrame) -> str:
    """
    Generates a natural language answer using Ollama based on the provided context DataFrame.
    """
    if context_df is None or context_df.empty:
        print("No context provided to generate answer.")
        return "I could not find any relevant information matching your query and filter."

    try:
        # Ensure the column name matches the output of your VectorStore._create_dataframe_from_results
        contents = context_df['content'].astype(str).tolist()
        context_string = "\n\n---\n\n".join(contents)
    except KeyError:
        print("Error: 'content' column not found in search results DataFrame.")
        return "Error retrieving content from search results to generate answer."
    except Exception as e:
        print(f"Error formatting context from DataFrame: {e}")
        return "Error processing search results before generating answer."

    # --- RAG Prompt Construction ---
    system_prompt = """You are a helpful assistant. Answer the user's question based *only* on the following context.
    If the context does not contain the answer, state clearly that you cannot answer from the provided information.
    Do not use any prior knowledge or information outside the given context. Be concise."""

    user_prompt = f"Context:\n{context_string}\n\n---\n\nQuestion: {question}"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    # --- End Prompt Construction ---

    try:
        print(f"Sending request to Ollama model: {OLLAMA_CHAT_MODEL}")
        # Call Ollama chat completion API
        response = ollama.chat(
            model=OLLAMA_CHAT_MODEL,
            messages=messages
            # Optional: Add stream=False if you don't want streaming
            # Optional: Add options={'temperature': 0.7} etc.
        )

        # Extract the answer content
        if response and 'message' in response and 'content' in response['message']:
            answer = response['message']['content']
            print("Received response from Ollama.")
            return answer.strip()
        else:
            print(f"Unexpected response structure from Ollama: {response}")
            return "Error: Could not get a valid response from the language model."

    except Exception as e:
        # Catch potential connection errors, model not found errors, etc.
        print(f"Error calling Ollama API: {e}")
        return f"Error interacting with the language model ({OLLAMA_CHAT_MODEL}): {e}"


def _generate_openai_answer(question: str, context_df: pd.DataFrame) -> str:
    """
    Generates a natural language answer using OpenAI based on the provided context DataFrame.
    """
    # Check if OpenAI client is available
    if not openai_client:
        return "Error: OpenAI API key not configured. Cannot generate answer."

    if context_df is None or context_df.empty:
        logging.warning("No context provided to generate answer.")
        # Ask OpenAI to respond naturally even without context? Or return fixed message?
        # For now, let's ask OpenAI directly based on the question alone if no context.
        context_string = "No relevant context found."
        # Alternatively: return "I could not find any relevant information..."
    else:
        try:
            contents = context_df['content'].astype(str).tolist()
            context_string = "\n\n---\n\n".join(contents)
        except KeyError:
            logging.error("Error: 'content' column not found in search results DataFrame.")
            return "Error retrieving content from search results to generate answer."
        except Exception as e:
            logging.error(f"Error formatting context from DataFrame: {e}", exc_info=True)
            return "Error processing search results before generating answer."

    # --- RAG Prompt Construction (same prompt works well) ---
    system_prompt = """You are a helpful assistant. Answer the user's question based *only* on the following context.
If the context does not contain the answer, state clearly that you cannot answer from the provided information.
Do not use any prior knowledge or information outside the given context. Be concise."""

    user_prompt = f"Context:\n{context_string}\n\n---\n\nQuestion: {question}"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    # --- End Prompt Construction ---

    try:
        logging.info(f"Sending request to OpenAI model: {OPENAI_CHAT_MODEL}")
        # --- Call OpenAI API ---
        response = openai_client.chat.completions.create(
            model=OPENAI_CHAT_MODEL,
            messages=messages,
            temperature=0.5,  # Adjust creativity/factuality
            max_tokens=500  # Limit response length
        )
        # --- End API Call ---

        # --- Extract Answer ---
        if response.choices and response.choices[0].message and response.choices[0].message.content:
            answer = response.choices[0].message.content
            logging.info("Received response from OpenAI.")
            return answer.strip()
        else:
            logging.error(f"Unexpected response structure from OpenAI: {response}")
            return "Error: Could not get a valid response from the language model."
        # --- End Extraction ---

    except openai.APIConnectionError as e:
        logging.error(f"OpenAI API request failed to connect: {e}", exc_info=True)
        return f"Error: Failed to connect to OpenAI API."
    except openai.RateLimitError as e:
        logging.error(f"OpenAI API request exceeded rate limit: {e}", exc_info=True)
        return f"Error: Rate limit exceeded for OpenAI API."
    except openai.AuthenticationError as e:
        logging.error(f"OpenAI API key invalid: {e}", exc_info=True)
        return f"Error: Invalid OpenAI API key."
    except openai.APIError as e:  # Catch generic API errors
        logging.error(f"OpenAI API returned an API Error: {e}", exc_info=True)
        return f"Error: OpenAI API returned an error ({e.status_code})."
    except Exception as e:
        logging.error(f"Unexpected error calling OpenAI API: {e}", exc_info=True)
        return f"Error interacting with the language model ({OPENAI_CHAT_MODEL})."


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
            _generate_openai_answer, request.question, search_results_df
        )
        # User for Ollama
        # final_answer = await run_sync_in_thread(
        #     _generate_ollama_answer, request.question, search_results_df
        # )

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
