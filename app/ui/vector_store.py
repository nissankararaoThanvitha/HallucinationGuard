import uuid

from langchain_chroma import Chroma
from app.embeddings.embedding_service import get_embedding_model


def create_session_vector_store(chunks):
    """
    Create a unique Chroma collection for the
    currently uploaded PDF.
    """

    embeddings = get_embedding_model()

    collection_name = (
        "uploaded_pdf_"
        + str(uuid.uuid4())
    )

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=collection_name,
    )

    return vector_store