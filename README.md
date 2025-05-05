# FastAPI RAG API with Timescale Vector and OpenAI

## Overview

This project implements a backend application using Python and FastAPI to provide a Retrieval-Augmented Generation (RAG) Question & Answering system. It allows users to ingest text-based documents, which are then chunked, embedded using an Ollama model (`nomic-embed-text`), and stored in a TimescaleDB database leveraging the Timescale Vector extension. Users can then ask questions related to the ingested documents, and the system retrieves relevant context from the database and generates a natural language answer using OpenAI's `gpt-4o-mini` model.

## Features

*   **Document Ingestion API (`POST /ingest/`)**: Accepts various text document uploads (PDF, DOCX, TXT, MD, etc.). Enforces a configurable file size limit. Chunks the documents, generates embeddings using Ollama, and stores chunks with metadata (`source_file`, `created_at`) in TimescaleDB.
*   **Document Listing API (`GET /documents/`)**: Lists the unique `source_file` names of all successfully ingested documents, allowing users to reference specific files in queries.
*   **Q&A API (`POST /query/`)**: Accepts a user question and an optional `source_file` filter.
    *   Retrieves relevant document chunks from TimescaleDB based on semantic similarity (using Ollama embeddings) and the optional filter.
    *   Generates a concise, context-aware answer using OpenAI (`gpt-4o-mini`) based *only* on the retrieved context.
*   **Dockerized:** Includes Dockerfile and Docker Compose setup for easy deployment of the application and the TimescaleDB database.
*   **Asynchronous:** Built with FastAPI for efficient handling of concurrent requests. Blocking operations (I/O, embedding, API calls) are handled in separate threads.

## Technology Stack

*   **Backend Framework:** FastAPI
*   **Web Server:** Uvicorn
*   **Database:** PostgreSQL with TimescaleDB + Timescale Vector / pgvector extensions
*   **Vector DB Client:** `timescale_vector` Python library
*   **Database Driver:** `psycopg2`
*   **Embedding Model:** Ollama (`nomic-embed-text` by default) - via `ollama` library
*   **Generation Model:** OpenAI (`gpt-4o-mini` by default) - via `openai` library
*   **Document Loading:** `langchain-community`, `unstructured`
*   **Text Splitting:** `langchain-text-splitters`
*   **Data Handling:** Pandas (primarily for DB interaction format)
*   **Configuration:** `python-dotenv`
*   **Containerization:** Docker, Docker Compose

## Project Structure

```
.
├── _temp/                  # Temporary file storage (mounted in Docker)
├── database/
│   └── vector_store.py     # Class managing Timescale Vector interactions
├── services/
│   ├── ingestion_service.py # Handles file loading, chunking, embedding, upserting
│   ├── qa_service.py        # Handles search and OpenAI answer generation
│   └── document_service.py  # Handles listing documents
├── config/
│   └── settings.py         # (Assumed by VectorStore) Settings management
├── .env                    # Local environment variables (DO NOT COMMIT)
├── .env.example            # Example environment variables template
├── config.py               # Application configuration loading
├── main.py                 # FastAPI application entrypoint
├── models.py               # Pydantic models for API requests/responses
├── requirements.txt        # Python dependencies
├── Dockerfile              # Docker build instructions for the FastAPI app
├── docker-compose.yml      # Docker Compose for running app + database
└── setup_database.py       # Script to initialize DB table/index
└── README.md               # This file
```

## Setup and Installation

### Prerequisites

*   Python 3.10+
*   Docker and Docker Compose
*   Ollama running locally (or accessible via network) with the embedding model pulled:
    ```bash
    ollama pull nomic-embed-text
    ```
*   An OpenAI API Key

### Steps

1.  **Clone the Repository:**
    ```bash
    git clone <your-repository-url>
    cd <repository-directory>
    ```

2.  **Configure Environment:**
    *   Copy the example environment file:
        ```bash
        cp .env.example .env
        ```
    *   Edit the `.env` file and add your actual `OPENAI_API_KEY`.
    *   Adjust database credentials (`POSTGRES_...`) and ports (`POSTGRES_PORT`, `APP_PORT`) if needed. The defaults align with the `docker-compose.yml`.

3.  **Install Python Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Start Database Service:**
    *   Use Docker Compose to start the TimescaleDB container in the background:
        ```bash
        cd docker
        docker-compose up -d timescaledb
        ```
    *   Wait a few moments for the database to initialize (check logs with `docker-compose logs -f timescaledb`).

5.  **Initialize Database Schema:**
    *   Run the setup script **once** to create the necessary table and index. This script uses the `DATABASE_URL` from your `.env` file if running locally, or connects appropriately if run within a potential setup container. *Ensure your local `.env` `DATABASE_URL` points to `localhost:POSTGRES_PORT` if running this script outside docker.*
        ```bash
        python setup_database.py
        ```
    *   *(Note: If running the full stack via `docker-compose up` later, you might integrate this step into an init container or entrypoint script for the `rag_api` service for better automation).*
    
6.  **(Optional) Generate and Ingest Test Data:**
    *   Make sure the API will be running (locally or via Docker) for the next step. If running locally, start it now: `uvicorn main:app --reload --port ${APP_PORT:-8000}`
    *   Run the script to generate fake documents and ingest them using the running API:
        ```bash
        python insert_test_data.py
        ```
    *   Stop the local Uvicorn server if you started it (`Ctrl+C`).
    
7.  **Run the FastAPI Application (Locally):**
    ```bash
    uvicorn main:app --reload --port ${APP_PORT:-8000}
    ```
    The API will be available at `http://localhost:${APP_PORT:-8000}`.

## API Usage

The API documentation is automatically generated by FastAPI and available via Swagger UI. Once the application is running (locally or via Docker), navigate to:

`http://localhost:${APP_PORT:-8000}/docs`

From there, you can explore and interact with the available endpoints:

*   **`POST /ingest/`**: Upload a document file for processing and storage.
*   **`GET /documents/`**: Retrieve a list of ingested source filenames.
*   **`POST /query/`**: Ask a question. Use the request body to provide the `question` and optionally filter by `source_file`.

**Example cURL Requests:**

```bash
# Ingest a document
curl -X POST -F 'file=@/path/to/your/document.pdf' http://localhost:8000/ingest/

# List ingested documents
curl -X GET http://localhost:8000/documents/

# Ask a question (no filter)
curl -X POST -H "Content-Type: application/json" -d '{"question": "What is the main topic of the document?"}' http://localhost:8000/query/

# Ask a question (filtered by source file)
curl -X POST -H "Content-Type: application/json" -d '{"question": "What was mentioned about project X?", "source_file": "document.pdf"}' http://localhost:8000/query/
```

## Testing

*(Showcase Requirement: Test Automation)*

## Testing

*(Showcase Requirement: Test Automation)*

Basic test automation is implemented using the `pytest` framework.

*   **Location:**
    *   `test_api.py`: Contains integration tests for the API endpoints, using FastAPI's `TestClient` (`httpx`) to simulate HTTP requests.
*   **Running Tests:**
    ```bash
    pytest -v
    ```
    *(Note: Integration tests might require a running database or further mocking of database interactions depending on their scope).*
*   **Coverage:** Test coverage is not explicitly measured yet, but the framework is in place to add comprehensive positive and negative test cases.
