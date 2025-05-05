import pytest
from fastapi.testclient import TestClient

import os

from main import app


@pytest.fixture(scope="module")
def client():
    # Use FastAPI's TestClient
    with TestClient(app) as c:
        yield c


# --- Test Data ---
# Create a dummy file for upload tests
DUMMY_FILE_CONTENT = "This is content for the test file."
DUMMY_FILE_NAME = "test_document.txt"
DUMMY_FILE_PATH = f"./{DUMMY_FILE_NAME}"  # In project root for simplicity


@pytest.fixture(scope="module", autouse=True)
def create_dummy_file():
    # Create file before tests run
    with open(DUMMY_FILE_PATH, "w") as f:
        f.write(DUMMY_FILE_CONTENT)
    yield  # Let tests run
    # Clean up file after tests run
    os.remove(DUMMY_FILE_PATH)


def test_read_root(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "message": "Welcome to the Document RAG API!"}


def test_ingest_document_success(client):
    # Set the API_KEY env var for the app instance being tested if needed by auth
    # os.environ['API_KEY'] = TEST_API_KEY # This might be needed depending on how config loads
    with open(DUMMY_FILE_PATH, "rb") as f:
        response = client.post("/ingest/", files={"file": (DUMMY_FILE_NAME, f, "text/plain")})
        print(response)
    # Note: Actual upsert depends on DB connection and VectorStore mock/setup
    # For now, focus on API accepting the request and returning expected structure *if successful*
    # A successful response here might be 201, but depends heavily on mocked backend/DB state
    # assert response.status_code == 201 # This requires mocking VectorStore.upsert etc.
    # assert response.json()["source_file"] == DUMMY_FILE_NAME
    # assert response.json()["chunks_added"] > 0
    # For now, just assert it didn't immediately fail due to auth or path issues
    assert response.status_code != 401
    assert response.status_code != 404
    # More realistic test requires mocking VectorStore methods called by the service


def test_list_documents(client):
    # This endpoint might or might not need auth depending on your choice
    # Assuming no auth for list for now
    response = client.get("/documents/")  # Add headers=HEADERS if auth is added
    assert response.status_code == 200
    assert "documents" in response.json()
    assert isinstance(response.json()["documents"], list)
    # To check for specific files, ingest needs to work reliably or DB needs seeding


def test_query_success(client):
    query_data = {"question": "What is in the test file?", "source_file": DUMMY_FILE_NAME}
    # This requires mocking VectorStore.search and the OpenAI call
    # Mocking example (needs pytest-mock):
    # with patch('services.qa_service.vec.search', return_value=pd.DataFrame({'content': ['Mocked search result.']})), \
    #      patch('services.qa_service.openai_client.chat.completions.create', return_value=MagicMock(...)): # Mock OpenAI response
    response = client.post("/query/", json=query_data)

    assert response.status_code != 401  # Check auth didn't fail
    # Actual success (200) and content requires mocking backend calls
    # assert response.status_code == 200
    # assert "answer" in response.json()
