from pathlib import Path

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
)


SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".docx"}


def load_document(file_path: str):
    """
    Load a single PDF, TXT, or DOCX file and return
    a list of LangChain Document objects.
    """

    path = Path(file_path)

    # Check whether the file exists
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    # Get the file extension
    extension = path.suffix.lower()

    # Check whether the file type is supported
    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type: {extension}. "
            f"Supported types are: {SUPPORTED_EXTENSIONS}"
        )

    # Select the correct loader
    if extension == ".pdf":
        loader = PyPDFLoader(str(path))

    elif extension == ".txt":
        loader = TextLoader(str(path), encoding="utf-8")

    elif extension == ".docx":
        loader = Docx2txtLoader(str(path))

    # Load the document
    documents = loader.load()

    return documents
def load_documents_from_folder(folder_path: str):
    """
    Load all supported documents from a folder.
    """

    folder = Path(folder_path)

    # Check whether the folder exists
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    # Store all loaded documents
    all_documents = []

    # Go through every file in the folder
    for file_path in folder.iterdir():

        # Skip folders
        if not file_path.is_file():
            continue

        # Check whether the file type is supported
        if file_path.suffix.lower() in SUPPORTED_EXTENSIONS:

            print(f"Loading: {file_path.name}")

            # Load the file and add its documents
            documents = load_document(str(file_path))

            all_documents.extend(documents)

    return all_documents