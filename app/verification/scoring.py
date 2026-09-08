def calculate_grounding_score(verification_results):
    """
    Calculate a confidence-weighted grounding score
    and an overall grounding verdict.

    This measures whether claims are supported by the
    retrieved evidence. It does not measure absolute truth.
    """

    if not verification_results:

        return {
            "grounding_score": 0.0,
            "overall_verdict": "NO VERIFIABLE CLAIMS",
            "supported": 0,
            "contradicted": 0,
            "no_evidence": 0,
            "total_claims": 0,
        }

    supported = 0
    contradicted = 0
    no_evidence = 0

    total_score = 0.0

    for result in verification_results:

        verdict = result["verdict"]

        confidence = float(
            result["confidence"]
        )

        if verdict == "SUPPORTED":

            supported += 1

            # Supported claims contribute according
            # to the model confidence.
            total_score += confidence

        elif verdict == "CONTRADICTED":

            contradicted += 1

            # Contradicted claims contribute zero.
            total_score += 0.0

        else:

            no_evidence += 1

            # NO_EVIDENCE claims contribute zero.
            total_score += 0.0

    total_claims = len(
        verification_results
    )

    grounding_score = (
        total_score / total_claims
    ) * 100

    grounding_score = round(
        grounding_score,
        2,
    )

    # Determine the overall verdict
    if grounding_score >= 80:

        overall_verdict = "HIGHLY GROUNDED"

    elif grounding_score >= 50:

        overall_verdict = "PARTIALLY GROUNDED"

    else:

        overall_verdict = "POORLY GROUNDED"

    return {
        "grounding_score": grounding_score,
        "overall_verdict": overall_verdict,
        "supported": supported,
        "contradicted": contradicted,
        "no_evidence": no_evidence,
        "total_claims": total_claims,
    }