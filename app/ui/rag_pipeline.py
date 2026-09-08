from app.generation.generator import get_llm


def create_context(results):
    """
    Convert retrieved documents into a context string.
    """

    context_parts = []

    for document, score in results:
        context_parts.append(
            document.page_content
        )

    return "\n\n".join(context_parts)


def generate_answer_from_pdf(
    question: str,
    vector_store,
    k: int = 3,
):
    """
    Answer a question using only the uploaded PDF.
    """

    # Step 1: Retrieve relevant chunks
    results = vector_store.similarity_search_with_score(
        query=question,
        k=k,
    )

    # Step 2: Create context
    context = create_context(results)

    # Step 3: Load Gemini
    llm = get_llm()

    # Step 4: Grounded prompt
    prompt = f"""
You are a helpful assistant answering questions using ONLY
the information provided in the context below.

Rules:
1. Use ONLY the provided context.
2. Do NOT use outside knowledge.
3. Do NOT invent information.
4. If the answer cannot be found in the context, say exactly:
"I don't have enough information in the uploaded PDF."
5. Keep the answer clear and concise.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""

    # Step 5: Generate answer
    response = llm.invoke(prompt)

    # Handle Gemini/LangChain response formats safely
    content = response.content

    if isinstance(content, str):
        answer = content

    elif isinstance(content, list):
        text_parts = []

        for item in content:
            if isinstance(item, dict) and "text" in item:
                text_parts.append(item["text"])

        answer = "".join(text_parts)

    else:
        answer = str(content)

    return answer, results