# setup_database.py
import logging
import os
import time
from database.vector_store import VectorStore  # Import your class
from config import settings  # Assuming your VectorStore uses config.settings

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Retry mechanism for database connection during startup
MAX_RETRIES = 5
RETRY_DELAY = 5  # seconds


def setup():
    vector_store = None
    for attempt in range(MAX_RETRIES):
        try:
            logging.info(f"Attempt {attempt + 1}/{MAX_RETRIES}: Initializing VectorStore...")
            # Ensure VectorStore uses environment variables for connection
            # If it reads from settings, ensure settings are loaded correctly
            vector_store = VectorStore()
            table_name = vector_store.vector_settings.table_name
            logging.info(f"Successfully initialized VectorStore for table '{table_name}'.")

            # Optional: Check connection explicitly if VectorStore init doesn't fail on bad connection
            # logging.info("Testing database connection...")
            # Test query, e.g., try listing tables or running a simple SELECT 1
            # vector_store.test_connection() # Needs implementation in VectorStore

            logging.info(f"Ensuring database table '{table_name}' exists...")
            vector_store.create_tables()  # Calls self.vec_client.create_tables()
            logging.info(f"Table '{table_name}' check/creation complete.")

            # Optional: Create index if it doesn't exist
            logging.info(f"Ensuring embedding index exists for '{table_name}'...")
            # Note: create_index might fail if called repeatedly without checks
            # Implement logic within VectorStore or here to check if index exists first
            # For simplicity now, just call it - timescale_vector might handle idempotency.
            vector_store.create_index()
            logging.info("Embedding index check/creation complete.")

            logging.info("Database setup successful!")
            return  # Exit loop on success

        except ImportError as ie:
            logging.error(
                f"Import error during setup: {ie}. Ensure all dependencies are installed ('pip install -r requirements.txt').")
            break  # Stop retrying if code/dependencies are missing
        except Exception as e:
            logging.warning(f"Database setup attempt {attempt + 1} failed: {e}")
            if attempt + 1 == MAX_RETRIES:
                logging.error("Max retries reached. Database setup failed.")
                raise  # Re-raise the last exception
            logging.info(f"Retrying in {RETRY_DELAY} seconds...")
            time.sleep(RETRY_DELAY)


if __name__ == "__main__":
    setup()
