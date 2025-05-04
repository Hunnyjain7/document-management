import json
import logging
import ollama
import time
from typing import Any, List, Optional, Tuple, Union
from datetime import datetime

import pandas as pd
import psycopg2

from config.settings import get_settings
from timescale_vector import client


class VectorStore:
    """A class for managing vector operations and database interactions."""

    def __init__(self):
        """Initialize the VectorStore with settings, OpenAI client, and Timescale Vector client."""
        self.settings = get_settings()
        # self.openai_client = OpenAI(api_key=self.settings.openai.api_key)
        # self.embedding_model = self.settings.openai.embedding_model
        self.vector_settings = self.settings.vector_store
        self.vec_client = client.Sync(
            self.settings.database.service_url,
            self.vector_settings.table_name,
            self.vector_settings.embedding_dimensions,
            time_partition_interval=self.vector_settings.time_partition_interval,
        )

    def get_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for the given text.

        Args:
            text: The input text to generate an embedding for.

        Returns:
            A list of floats representing the embedding.
        """
        text = text.replace("\n", " ")
        start_time = time.time()

        response = ollama.embeddings(model='nomic-embed-text', prompt=text)
        embedding = response['embedding']

        elapsed_time = time.time() - start_time
        logging.info(f"Embedding generated in {elapsed_time:.3f} seconds")
        return embedding

    def create_tables(self) -> None:
        """Create the necessary tablesin the database"""
        self.vec_client.create_tables()

    def create_index(self) -> None:
        """Create the StreamingDiskANN index to spseed up similarity search"""
        self.vec_client.create_embedding_index(client.DiskAnnIndex())

    def drop_index(self) -> None:
        """Drop the StreamingDiskANN index in the database"""
        self.vec_client.drop_embedding_index()

    # def upsert(self, df: pd.DataFrame) -> None:
    #     """
    #     Insert or update records in the database from a pandas DataFrame.
    #
    #     Args:
    #         df: A pandas DataFrame containing the data to insert or update.
    #             Expected columns: id, metadata, contents, embedding
    #     """
    #     records = df.to_records(index=False)
    #     self.vec_client.upsert(list(records))
    #     logging.info(
    #         f"Inserted {len(df)} records into {self.vector_settings.table_name}"
    #     )

    def upsert(self, df: pd.DataFrame) -> None:
        """
        Insert or update records in the database from a pandas DataFrame.
        Converts metadata dictionaries to JSON strings before upserting.

        Args:
            df: A pandas DataFrame containing the data to insert or update.
                Expected columns: id, metadata, contents, embedding
        """
        records_for_upsert = []
        required_columns = {'id', 'metadata', 'contents', 'embedding'}
        if not required_columns.issubset(df.columns):
            raise ValueError(
                f"DataFrame missing required columns for upsert. Need: {required_columns}, Got: {list(df.columns)}")

        logging.debug(f"Preparing {len(df)} records for upsert...")
        # Iterate through DataFrame rows to create tuples with JSON string metadata
        for row_tuple in df.itertuples(index=False):
            try:
                # Reconstruct the tuple in the expected order, converting metadata
                # Assuming the order is id, metadata, contents, embedding based on common patterns
                # Double-check this order if upsert still fails.
                record_id = getattr(row_tuple, 'id')
                metadata_dict = getattr(row_tuple, 'metadata')
                contents = getattr(row_tuple, 'contents')
                embedding = getattr(row_tuple, 'embedding')

                # Convert metadata dict to JSON string
                metadata_json = json.dumps(metadata_dict)

                # Create the tuple in the correct order for the DB insert
                # Ensure this order matches the columns in your Timescale Vector table
                # and the order expected by vec_client.upsert's internal SQL
                prepared_tuple = (record_id, metadata_json, contents, embedding)
                records_for_upsert.append(prepared_tuple)

            except AttributeError as e:
                logging.error(f"Error processing DataFrame row for upsert: {e}. Row: {row_tuple}")
                # Decide: skip row or raise error? Skipping for now.
                continue
            except TypeError as e:
                logging.error(f"Error serializing metadata to JSON: {e}. Metadata: {metadata_dict}")
                # Decide: skip row or raise error? Skipping for now.
                continue

        if not records_for_upsert:
            logging.warning("No records were successfully prepared for upsert after processing.")
            return

        logging.debug(f"Calling vec_client.upsert with {len(records_for_upsert)} prepared records.")
        try:
            # Pass the list of tuples with JSON strings to the client library
            self.vec_client.upsert(records_for_upsert)  # Pass the processed list
            logging.info(
                f"Successfully upserted {len(records_for_upsert)} records into {self.vector_settings.table_name}"
            )
        except Exception as e:
            logging.error(f"vec_client.upsert failed: {e}", exc_info=True)
            # Re-raise or handle as appropriate
            raise

    def search(
            self,
            query_text: str,
            limit: int = 5,
            distance_threshold: float = 0.5,
            metadata_filter: Union[dict, List[dict]] = None,
            predicates: Optional[client.Predicates] = None,
            time_range: Optional[Tuple[datetime, datetime]] = None,
            return_dataframe: bool = True,
    ) -> Union[List[Tuple[Any, ...]], pd.DataFrame]:
        """
        Query the vector database for similar embeddings based on input text.

        More info:
            https://github.com/timescale/docs/blob/latest/ai/python-interface-for-pgvector-and-timescale-vector.md

        Args:
            query_text: The input text to search for.
            limit: The maximum number of results to return.
            metadata_filter: A dictionary or list of dictionaries for equality-based metadata filtering.
            predicates: A Predicates object for complex metadata filtering.
                - Predicates objects are defined by the name of the metadata key, an operator, and a value.
                - Operators: ==, !=, >, >=, <, <=
                - & is used to combine multiple predicates with AND operator.
                - | is used to combine multiple predicates with OR operator.
            time_range: A tuple of (start_date, end_date) to filter results by time.
            return_dataframe: Whether to return results as a DataFrame (default: True).

        Returns:
            Either a list of tuples or a pandas DataFrame containing the search results.

        Basic Examples:
            Basic search:
                vector_store.search("What are your shipping options?")
            Search with metadata filter:
                vector_store.search("Shipping options", metadata_filter={"category": "Shipping"})

        Predicates Examples:
            Search with predicates:
                vector_store.search("Pricing", predicates=client.Predicates("price", ">", 100))
            Search with complex combined predicates:
                complex_pred = (client.Predicates("category", "==", "Electronics") & client.Predicates("price", "<", 1000)) | \
                               (client.Predicates("category", "==", "Books") & client.Predicates("rating", ">=", 4.5))
                vector_store.search("High-quality products", predicates=complex_pred)

        Time-based filtering:
            Search with time range:
                vector_store.search("Recent updates", time_range=(datetime(2024, 1, 1), datetime(2024, 1, 31)))
        """
        query_embedding = self.get_embedding(query_text)

        start_time = time.time()

        search_args = {
            "limit": limit,
        }

        if metadata_filter:
            search_args["filter"] = metadata_filter

        if predicates:
            search_args["predicates"] = predicates

        if time_range:
            start_date, end_date = time_range
            search_args["uuid_time_filter"] = client.UUIDTimeRange(start_date, end_date)

        results = self.vec_client.search(query_embedding, **search_args)
        print("results", results)
        if distance_threshold is not None:
            results = [result for result in results if result[-1] <= distance_threshold]
        elapsed_time = time.time() - start_time

        logging.info(f"Vector search completed in {elapsed_time:.3f} seconds")

        if return_dataframe:
            return self._create_dataframe_from_results(results)
        else:
            return results

    def _create_dataframe_from_results(
            self,
            results: List[Tuple[Any, ...]],
    ) -> pd.DataFrame:
        """
        Create a pandas DataFrame from the search results.

        Args:
            results: A list of tuples containing the search results.

        Returns:
            A pandas DataFrame containing the formatted search results.
        """
        # Convert results to DataFrame
        df = pd.DataFrame(
            results, columns=["id", "metadata", "content", "embedding", "distance"]
        )

        # Expand metadata column
        df = pd.concat(
            [df.drop(["metadata"], axis=1), df["metadata"].apply(pd.Series)], axis=1
        )

        # Convert id to string for better readability
        df["id"] = df["id"].astype(str)

        return df

    def delete(
            self,
            ids: List[str] = None,
            metadata_filter: dict = None,
            delete_all: bool = False,
    ) -> None:
        """Delete records from the vector database.

        Args:
            ids (List[str], optional): A list of record IDs to delete.
            metadata_filter (dict, optional): A dictionary of metadata key-value pairs to filter records for deletion.
            delete_all (bool, optional): A boolean flag to delete all records.

        Raises:
            ValueError: If no deletion criteria are provided or if multiple criteria are provided.

        Examples:
            Delete by IDs:
                vector_store.delete(ids=["8ab544ae-766a-11ef-81cb-decf757b836d"])

            Delete by metadata filter:
                vector_store.delete(metadata_filter={"category": "Shipping"})

            Delete all records:
                vector_store.delete(delete_all=True)
        """
        if sum(bool(x) for x in (ids, metadata_filter, delete_all)) != 1:
            raise ValueError(
                "Provide exactly one of: ids, metadata_filter, or delete_all"
            )

        if delete_all:
            self.vec_client.delete_all()
            logging.info(f"Deleted all records from {self.vector_settings.table_name}")
        elif ids:
            self.vec_client.delete_by_ids(ids)
            logging.info(
                f"Deleted {len(ids)} records from {self.vector_settings.table_name}"
            )
        elif metadata_filter:
            self.vec_client.delete_by_metadata(metadata_filter)
            logging.info(
                f"Deleted records matching metadata filter from {self.vector_settings.table_name}"
            )

    def list_distinct_metadata_values(self, field_name: str) -> List[str]:
        """
        Retrieves a sorted list of unique non-null values for a given metadata field
        from the underlying Timescale Vector table using a direct DB connection.

        Args:
            field_name: The key within the metadata JSONB to retrieve distinct values for (e.g., 'source_file').

        Returns:
            A list of unique string values for the specified field.
        """
        results = []
        conn = None  # Initialize connection variable
        cursor = None # Initialize cursor variable

        # --- Use direct psycopg2 connection ---
        try:
            # Get connection string from settings used during __init__
            conn_string = self.settings.database.service_url
            if not conn_string:
                 logging.error("VectorStore: Database service URL is not configured in settings.")
                 raise ValueError("Database connection string is missing.")

            logging.debug(f"Connecting directly to database for distinct metadata query...")
            conn = psycopg2.connect(conn_string)
            cursor = conn.cursor()

            # Assumes metadata is stored in a JSONB column named 'metadata'
            # Assumes table name is stored in self.vector_settings.table_name
            table_name = self.vector_settings.table_name # Get table name from settings
            query = f"""
                SELECT DISTINCT metadata->>%s
                FROM "{table_name}" -- Ensure table name is quoted if needed
                WHERE metadata->>%s IS NOT NULL
                ORDER BY 1;
            """
            logging.debug(f"Executing distinct metadata query: {query % (field_name, field_name)}")
            cursor.execute(query, (field_name, field_name)) # Pass field_name as parameter twice
            fetched_results = cursor.fetchall()
            results = [row[0] for row in fetched_results] # Extract the first column

            logging.info(f"Found {len(results)} distinct values for metadata field '{field_name}' in table '{table_name}'")

        except psycopg2.Error as db_err: # Catch specific psycopg2 errors
            logging.error(f"Database error fetching distinct metadata for '{field_name}' from table '{table_name}': {db_err}", exc_info=True)
            raise RuntimeError(f"Database error while fetching distinct metadata for '{field_name}'.") from db_err
        except Exception as e:
            logging.error(f"Unexpected error fetching distinct metadata for '{field_name}' from table '{table_name}': {e}", exc_info=True)
            raise RuntimeError(f"Unexpected error while fetching distinct metadata for '{field_name}'.") from e
        finally:
            # Ensure cursor and connection are closed even if errors occur
            if cursor:
                cursor.close()
            if conn:
                conn.close()
            logging.debug("Direct database connection closed.")
        # --- End direct connection block ---

        return results

