import os
import random
import time
import requests  # For making HTTP requests
from faker import Faker
from dotenv import load_dotenv

# --- Configuration ---
load_dotenv()  # Load variables from .env file

# Get API details from environment variables
API_BASE_URL = f"http://localhost:{os.getenv('APP_PORT', '8000')}"  # Default to 8000
INGEST_ENDPOINT = f"{API_BASE_URL}/ingest/"

# Data generation settings
NUM_DOCUMENTS_TO_INSERT = 10  # How many fake documents to create and ingest
MIN_PARAGRAPHS = 5
MAX_PARAGRAPHS = 15
TEMP_DIR = "_generated_temp_data"  # Directory to temporarily store generated files

fake = Faker()


# --- End Configuration ---

def generate_text_content(num_paragraphs):
    """Generates random text paragraphs using Faker."""
    return "\n\n".join(fake.paragraph(nb_sentences=random.randint(4, 8)) for _ in range(num_paragraphs))


def ingest_document_via_api(file_path, file_name):
    """Sends a document file to the /ingest/ API endpoint."""
    print(f"  Attempting to ingest '{file_name}' via API...")
    try:
        with open(file_path, 'rb') as f:
            files = {'file': (file_name, f, 'text/plain')}  # 'text/plain' is suitable for .txt
            response = requests.post(INGEST_ENDPOINT, files=files, timeout=60)  # Added timeout

        if response.status_code == 201:
            response_data = response.json()
            print(
                f"  Success: Ingested '{response_data.get('source_file', file_name)}'. Chunks added: {response_data.get('chunks_added', 'N/A')}")
            return True
        else:
            error_detail = "No detail provided."
            try:
                error_detail = response.json().get('detail', response.text)
            except requests.exceptions.JSONDecodeError:
                error_detail = response.text
            print(f"  Failed: Status code {response.status_code}. Detail: {error_detail}")
            return False

    except requests.exceptions.RequestException as e:
        print(f"  Error: API request failed for '{file_name}'. Exception: {e}")
        return False
    except Exception as e:
        print(f"  Error: An unexpected error occurred during ingestion of '{file_name}'. Exception: {e}")
        return False


def create_and_ingest_files():
    """Generates test files and calls the API to ingest them."""
    if not os.path.exists(TEMP_DIR):
        os.makedirs(TEMP_DIR)
        print(f"Created temporary directory: {TEMP_DIR}")

    print(f"\nStarting generation and ingestion of {NUM_DOCUMENTS_TO_INSERT} documents...")
    successful_ingestions = 0

    for i in range(NUM_DOCUMENTS_TO_INSERT):
        print("-" * 20)
        doc_number = i + 1
        num_paragraphs = random.randint(MIN_PARAGRAPHS, MAX_PARAGRAPHS)
        content = generate_text_content(num_paragraphs)
        file_name = f"generated_doc_{doc_number:03d}.txt"  # e.g., generated_doc_001.txt
        temp_file_path = os.path.join(TEMP_DIR, file_name)

        print(
            f"Document {doc_number}/{NUM_DOCUMENTS_TO_INSERT}: Generating '{file_name}' ({num_paragraphs} paragraphs)...")

        # 1. Write content to temporary file
        try:
            with open(temp_file_path, "w", encoding="utf-8") as f:
                f.write(content)
        except IOError as e:
            print(f"  Error: Failed to write temporary file {temp_file_path}: {e}")
            continue  # Skip to next document

        # 2. Ingest the temporary file via API
        success = ingest_document_via_api(temp_file_path, file_name)
        if success:
            successful_ingestions += 1

        # 3. Clean up the temporary file
        try:
            os.remove(temp_file_path)
        except OSError as e:
            print(f"  Warning: Failed to remove temporary file {temp_file_path}: {e}")

        # Optional: Add a small delay between requests
        time.sleep(0.5)  # Sleep for 500ms

    # 4. Clean up the temporary directory if empty (optional)
    try:
        if not os.listdir(TEMP_DIR):
            os.rmdir(TEMP_DIR)
            print(f"\nRemoved empty temporary directory: {TEMP_DIR}")
        else:
            print(f"\nNote: Temporary directory '{TEMP_DIR}' may contain files if errors occurred.")
    except OSError as e:
        print(f"\nWarning: Failed to remove temporary directory {TEMP_DIR}: {e}")

    print("-" * 20)
    print(
        f"\nIngestion process complete. Successfully ingested {successful_ingestions}/{NUM_DOCUMENTS_TO_INSERT} documents.")


if __name__ == "__main__":
    create_and_ingest_files()
