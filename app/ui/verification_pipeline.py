import json
import time

from concurrent.futures import ThreadPoolExecutor

from app.verification.claim_extractor import extract_claims_llm
from app.generation.generator import get_llm


VALID_VERDICTS = {
    "SUPPORTED",
    "CONTRADICTED",
    "NO_EVIDENCE",
}


def verify_claims_with_llm(claim_evidence_pairs):
    """
    Verify multiple claims against their retrieved evidence
    using a single Gemini call.
    """

    llm = get_llm()

    verification_input = []

    for index, item in enumerate(
        claim_evidence_pairs,
        start=1,
    ):
        verification_input.append(
            f"""
CLAIM {index}:
{item["claim"]}

EVIDENCE FOR CLAIM {index}:
{item["evidence"]}
"""
        )

    all_claims_text = "\n".join(
        verification_input
    )

    prompt = f"""
You are a strict factual verification system.

You must verify each CLAIM using ONLY the
EVIDENCE provided for that claim.

There are exactly three possible verdicts:

SUPPORTED:
The evidence directly states or clearly entails the claim.

CONTRADICTED:
The evidence explicitly conflicts with the claim.

NO_EVIDENCE:
The evidence does not clearly support or contradict the claim.

IMPORTANT RULES:

1. Use ONLY the provided evidence.
2. Do NOT use outside knowledge.
3. Do NOT make assumptions.
4. Do NOT treat absence of information as contradiction.
5. Similar wording alone is not enough for support.
6. Paraphrases are allowed when their meaning is clearly equivalent.
7. The evidence must actually establish the claim.
8. If the evidence is insufficient, use NO_EVIDENCE.
9. Evaluate every claim independently.
10. Do not let evidence for one claim influence another claim.
11. Return exactly one result for every claim.
12. Keep the original claim number.
13. Confidence must be a number between 0 and 1.
14. Return ONLY valid JSON.
15. Do not return markdown or explanations outside the JSON.

Return exactly this format:

[
    {{
        "claim_id": 1,
        "verdict": "SUPPORTED",
        "confidence": 0.95,
        "reason": "Brief explanation based only on the evidence."
    }},
    {{
        "claim_id": 2,
        "verdict": "NO_EVIDENCE",
        "confidence": 0.90,
        "reason": "The evidence does not clearly establish the claim."
    }}
]

CLAIMS AND EVIDENCE:

{all_claims_text}
"""

    # --------------------------------------------------
    # GEMINI VERIFICATION TIMER
    # --------------------------------------------------

    verification_start = time.time()

    response = llm.invoke(prompt)

    verification_time = (
        time.time() - verification_start
    )

    print(
        f"\nGEMINI VERIFICATION TIME: "
        f"{verification_time:.2f} seconds"
    )

    # --------------------------------------------------
    # PROCESS RESPONSE
    # --------------------------------------------------

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

    # Remove accidental markdown code fences

    if text.startswith("```json"):
        text = text[7:]

    elif text.startswith("```"):
        text = text[3:]

    if text.endswith("```"):
        text = text[:-3]

    text = text.strip()

    # --------------------------------------------------
    # PARSE JSON
    # --------------------------------------------------

    try:

        results = json.loads(text)

    except json.JSONDecodeError as error:

        raise ValueError(
            f"Failed to parse Gemini verification response.\n"
            f"LLM response: {text}"
        ) from error

    if not isinstance(results, list):

        raise ValueError(
            "Gemini verification response must be a JSON list."
        )

    return results


def retrieve_evidence_for_claim(
    claim,
    vector_store,
    k,
):
    """
    Retrieve evidence for a single claim.

    This function is designed to run in parallel.
    """

    results = vector_store.similarity_search_with_score(
        query=claim,
        k=k,
    )

    evidence_parts = []

    retrieved_chunks = []

    for document, score in results:

        evidence_parts.append(
            document.page_content
        )

        retrieved_chunks.append(
            {
                "text": document.page_content,
                "distance": float(score),
                "metadata": document.metadata,
            }
        )

    evidence = "\n\n".join(
        evidence_parts
    )

    return {
        "claim": claim,
        "evidence": evidence,
        "retrieved_chunks": retrieved_chunks,
    }


def verify_answer(
    answer: str,
    vector_store,
    k: int = 1,
):
    """
    Extract claims, retrieve evidence for each claim
    in parallel, and verify all claims using ONE Gemini call.
    """

    total_start = time.time()

    # --------------------------------------------------
    # CHECK EMPTY ANSWER
    # --------------------------------------------------

    if (
        answer.strip()
        == "I don't have enough information in the uploaded PDF."
    ):
        return []

    # --------------------------------------------------
    # DEBUG: ANSWER SIZE
    # --------------------------------------------------

    print(
        f"\nANSWER CHARACTERS: "
        f"{len(answer)}"
    )

    print(
        f"ANSWER WORDS: "
        f"{len(answer.split())}"
    )

    # --------------------------------------------------
    # STEP 1: EXTRACT CLAIMS
    # --------------------------------------------------

    claim_start = time.time()

    claims = extract_claims_llm(answer)

    claim_extraction_time = (
        time.time() - claim_start
    )

    print(
        f"\nCLAIM EXTRACTION TIME: "
        f"{claim_extraction_time:.2f} seconds"
    )

    print(
        f"NUMBER OF CLAIMS: "
        f"{len(claims)}"
    )

    if not claims:
        return []

    # --------------------------------------------------
    # STEP 2: RETRIEVE EVIDENCE IN PARALLEL
    # --------------------------------------------------

    retrieval_start = time.time()

    with ThreadPoolExecutor(
        max_workers=min(8, len(claims))
    ) as executor:

        futures = [
            executor.submit(
                retrieve_evidence_for_claim,
                claim,
                vector_store,
                k,
            )
            for claim in claims
        ]

        claim_evidence_pairs = [
            future.result()
            for future in futures
        ]

    retrieval_time = (
        time.time() - retrieval_start
    )

    print(
        f"\nRETRIEVAL TIME: "
        f"{retrieval_time:.2f} seconds"
    )

    # --------------------------------------------------
    # CALCULATE EVIDENCE SIZE
    # --------------------------------------------------

    total_evidence_characters = sum(
        len(item["evidence"])
        for item in claim_evidence_pairs
    )

    print(
        f"TOTAL EVIDENCE CHARACTERS SENT TO GEMINI: "
        f"{total_evidence_characters}"
    )

    # --------------------------------------------------
    # STEP 3: ONE GEMINI CALL FOR ALL CLAIMS
    # --------------------------------------------------

    llm_results = verify_claims_with_llm(
        claim_evidence_pairs
    )

    # --------------------------------------------------
    # STEP 4: MATCH GEMINI RESULTS TO CLAIMS
    # --------------------------------------------------

    verification_results = []

    result_by_id = {}

    for result in llm_results:

        claim_id = result.get(
            "claim_id"
        )

        if claim_id is None:
            continue

        result_by_id[int(claim_id)] = result

    for index, item in enumerate(
        claim_evidence_pairs,
        start=1,
    ):

        result = result_by_id.get(index)

        if result is None:

            raise ValueError(
                f"Gemini did not return a result "
                f"for claim {index}."
            )

        verdict = result.get(
            "verdict"
        )

        if verdict not in VALID_VERDICTS:

            raise ValueError(
                f"Invalid verdict returned by Gemini: "
                f"{verdict}"
            )

        confidence = float(
            result.get(
                "confidence",
                0.0,
            )
        )

        confidence = max(
            0.0,
            min(
                1.0,
                confidence,
            ),
        )

        reason = result.get(
            "reason",
            "",
        )

        verification_result = {
            "claim": item["claim"],
            "evidence": item["evidence"],
            "retrieved_chunks": item[
                "retrieved_chunks"
            ],
            "verdict": verdict,
            "confidence": confidence,
            "reason": reason,
        }

        verification_results.append(
            verification_result
        )

        # --------------------------------------------------
        # TERMINAL DEBUGGING
        # --------------------------------------------------

        print(
            "\n" + "=" * 70
        )

        print("CLAIM:")

        print(
            item["claim"]
        )

        print(
            "\nGEMINI VERIFICATION:"
        )

        print(
            f"Verdict: {verdict}"
        )

        print(
            f"Confidence: {confidence:.4f}"
        )

        print(
            f"Reason: {reason}"
        )

        print(
            "=" * 70
        )

    # --------------------------------------------------
    # TOTAL TIME
    # --------------------------------------------------

    total_time = (
        time.time() - total_start
    )

    print(
        f"\nTOTAL VERIFICATION TIME: "
        f"{total_time:.2f} seconds"
    )

    return verification_results