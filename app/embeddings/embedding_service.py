from langchain_huggingface import HuggingFaceEmbeddings


def get_embedding_model():
    """
    Create and return the embedding model.
    """

    model_name = "sentence-transformers/all-MiniLM-L6-v2"

    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={
            "device": "cpu"
        },
        encode_kwargs={
            "normalize_embeddings": True
        }
    )

    return embeddings