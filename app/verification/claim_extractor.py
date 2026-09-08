import json

from app.generation.generator import get_llm


def extract_claims_llm(answer: str) -> list[str]:
    """
    Extract atomic factual claims from the generated answer.

    Claims must preserve exactly what the answer states.
    The extractor must not create new relationships,
    interpretations, or assumptions.
    """

    llm = get_llm()

    prompt = f"""
You are a strict factual claim extraction system.

Your task is to extract atomic factual claims from the
given answer.

IMPORTANT:

A claim must represent something that is explicitly stated
or directly expressed in the answer.

Do NOT add information that is not explicitly stated.

Rules:

1. Split genuinely independent factual statements into
   separate atomic claims.

2. Preserve the original meaning of every claim.

3. Preserve the original subject of the statement.

4. Do NOT create new relationships between concepts.

5. Do NOT interpret one item as a component, cause,
   definition, consequence, or subset of another item unless
   the answer explicitly states that relationship.

6. Do NOT add assumptions or outside knowledge.

7. Do NOT convert two things connected by "and" into a
   parent-child relationship.

8. Be especially careful with lists containing "or".

   If the answer says:

   "Experience with academic, personal, or open-source tools"

   do NOT create three separate claims:

   "Experience with academic tools."
   "Experience with personal tools."
   "Experience with open-source tools."

   Instead preserve the original meaning as ONE claim:

   "Experience with academic, personal, or open-source tools."

9. When several items are explicitly presented as alternatives
   using "or", preserve them together in one claim unless the
   answer clearly states that each item is independently required.

10. When several items are explicitly presented as a combined
    requirement using "and", split them only when each item
    remains an independently verifiable statement without
    changing the meaning.

11. Do not turn a general statement into several stronger
    statements.

12. Keep claims as close as possible to the wording of the answer.

13. Ignore greetings, opinions, and conversational filler.

14. Do not rewrite claims to make them stronger or weaker.

15. Return ONLY a valid JSON array of strings.

16. Do not return markdown.

17. Do not return explanations.

Example 1:

Answer:
"The role requires attention to detail and good communication skills."

Output:
[
    "The role requires attention to detail.",
    "The role requires good communication skills."
]

Example 2:

Answer:
"The role requires experience with academic, personal, or
open-source tools."

Output:
[
    "The role requires experience with academic, personal, or open-source tools."
]

Example 3:

Answer:
"The role requires attention to detail and the ability to
spot issues."

Output:
[
    "The role requires attention to detail.",
    "The role requires the ability to spot issues."
]

Example 4:

Answer:
"The role requires experience with Python, Java, or C++."

Output:
[
    "The role requires experience with Python, Java, or C++."
]

Example 5:

Answer:
"Good communication skills are important. You should also
be comfortable asking questions."

Output:
[
    "Good communication skills are important.",
    "You should be comfortable asking questions."
]

Now extract the claims from this answer:

{answer}
"""

    response = llm.invoke(prompt)

    content = response.content

    if isinstance(content, str):
        text = content

    elif isinstance(content, list):
        text_parts = []

        for item in content:
            if isinstance(item, dict) and "text" in item:
                text_parts.append(item["text"])

        text = "".join(text_parts)

    else:
        text = str(content)

    text = text.strip()

    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]

    if text.endswith("```"):
        text = text[:-3]

    text = text.strip()

    try:
        claims = json.loads(text)

    except json.JSONDecodeError as error:
        raise ValueError(
            f"Failed to parse claims as JSON.\n"
            f"LLM response: {text}"
        ) from error

    if not isinstance(claims, list):
        raise ValueError(
            "LLM did not return a JSON list."
        )

    claims = [
        str(claim).strip()
        for claim in claims
        if str(claim).strip()
    ]

    return claims