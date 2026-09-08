import os
import tempfile

from app.ingestion.loader import load_document
from app.processing.chunker import chunk_documents


def process_uploaded_pdf(uploaded_file):
    """
    Save the uploaded PDF temporarily,
    load it, and split it into chunks.
    """

    # Create a temporary PDF file
    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".pdf",
    ) as temp_file:

        temp_file.write(
            uploaded_file.getbuffer()
        )

        temp_path = temp_file.name

    try:
        # Load PDF
        documents = load_document(
            temp_path
        )

        # Split PDF into chunks
        chunks = chunk_documents(
            documents
        )

        return chunks

    finally:
        # Delete temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)