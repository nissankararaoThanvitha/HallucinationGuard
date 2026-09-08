import streamlit as st

from app.ui.pdf_processor import process_uploaded_pdf
from app.ui.vector_store import create_session_vector_store
from app.ui.rag_pipeline import generate_answer_from_pdf
from app.ui.verification_pipeline import verify_answer
from app.verification.scoring import calculate_grounding_score


# ==================================================
# PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="HallucinationGuard",
    page_icon="🛡️",
    layout="wide",
)


# ==================================================
# CUSTOM STYLING
# ==================================================

st.markdown(
    """
    <style>

    /* Main title */
    .main-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 18px;
        color: #666666;
        margin-bottom: 25px;
    }

    /* Section headings */
    .section-title {
        font-size: 25px;
        font-weight: 650;
        margin-top: 10px;
        margin-bottom: 10px;
    }

    /* Information card */
    .info-card {
        padding: 18px;
        border-radius: 10px;
        border: 1px solid #dddddd;
        margin-bottom: 15px;
    }

    /* Claim card */
    .claim-card {
        padding: 16px;
        border-radius: 10px;
        border: 1px solid #dddddd;
        margin-bottom: 12px;
    }

    /* Small label */
    .small-label {
        font-size: 13px;
        color: #777777;
        margin-bottom: 4px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ==================================================
# HEADER
# ==================================================

st.markdown(
    '<div class="main-title">🛡️ HallucinationGuard</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="subtitle">
    An evidence-grounded RAG system that answers questions from
    your document and verifies the factual claims in its response.
    </div>
    """,
    unsafe_allow_html=True,
)


# ==================================================
# SESSION STATE
# ==================================================

if "pdf_processed" not in st.session_state:
    st.session_state["pdf_processed"] = False

if "vector_store" not in st.session_state:
    st.session_state["vector_store"] = None

if "file_name" not in st.session_state:
    st.session_state["file_name"] = None


# ==================================================
# STEP 1 — UPLOAD DOCUMENT
# ==================================================

st.markdown(
    '<div class="section-title">📄 1. Upload Document</div>',
    unsafe_allow_html=True,
)

uploaded_file = st.file_uploader(
    "Upload a PDF to begin",
    type=["pdf"],
)


# ==================================================
# PROCESS PDF
# ==================================================

if uploaded_file is not None:

    st.markdown(
        f"""
        <div class="info-card">
            <div class="small-label">Selected document</div>
            <strong>{uploaded_file.name}</strong>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button(
        "⚙️ Process PDF",
        type="primary",
        use_container_width=True,
    ):

        with st.spinner(
            "Processing PDF and creating vector database..."
        ):

            try:

                # ----------------------------------
                # PROCESS PDF
                # ----------------------------------

                chunks = process_uploaded_pdf(
                    uploaded_file
                )

                # ----------------------------------
                # CREATE SESSION VECTOR STORE
                # ----------------------------------

                vector_store = (
                    create_session_vector_store(
                        chunks
                    )
                )

                # ----------------------------------
                # SAVE SESSION STATE
                # ----------------------------------

                st.session_state[
                    "vector_store"
                ] = vector_store

                st.session_state[
                    "pdf_processed"
                ] = True

                st.session_state[
                    "file_name"
                ] = uploaded_file.name

                st.success(
                    f"✅ PDF processed successfully — "
                    f"{len(chunks)} chunks created."
                )

            except Exception as error:

                st.session_state[
                    "pdf_processed"
                ] = False

                st.error(
                    f"Error processing PDF: {error}"
                )


# ==================================================
# QUESTION SECTION
# ==================================================

if st.session_state.get(
    "pdf_processed",
    False,
):

    st.divider()

    st.markdown(
        '<div class="section-title">💬 2. Ask a Question</div>',
        unsafe_allow_html=True,
    )

    st.caption(
        f"📄 Answering only from: "
        f"**{st.session_state['file_name']}**"
    )

    question = st.text_input(
        "Your question",
        placeholder=(
            "Example: What skills are required for this role?"
        ),
    )

    ask_button = st.button(
        "🔍 Ask Question",
        type="primary",
        use_container_width=True,
    )


    # ==================================================
    # ASK QUESTION
    # ==================================================

    if ask_button:

        if not question.strip():

            st.warning(
                "Please enter a question."
            )

        else:

            try:

                # ==========================================
                # GENERATE ANSWER
                # ==========================================

                with st.spinner(
                    "Searching the document and generating an answer..."
                ):

                    answer, results = (
                        generate_answer_from_pdf(
                            question=question,
                            vector_store=(
                                st.session_state[
                                    "vector_store"
                                ]
                            ),
                        )
                    )


                # ==========================================
                # ANSWER
                # ==========================================

                st.divider()

                st.markdown(
                    '<div class="section-title">🤖 3. Generated Answer</div>',
                    unsafe_allow_html=True,
                )

                st.info(answer)


                # ==========================================
                # RETRIEVED EVIDENCE
                # ==========================================

                with st.expander(
                    "📚 View Retrieved Evidence",
                ):

                    st.caption(
                        "These document chunks were retrieved "
                        "to generate the answer."
                    )

                    for index, (
                        document,
                        score,
                    ) in enumerate(
                        results,
                        start=1,
                    ):

                        st.markdown(
                            f"**Evidence {index}**"
                        )

                        st.write(
                            document.page_content
                        )

                        st.caption(
                            f"Chroma distance: {score:.4f}"
                        )

                        if index < len(results):

                            st.divider()


                # ==========================================
                # HALLUCINATION VERIFICATION
                # ==========================================

                st.divider()

                st.markdown(
                    '<div class="section-title">🛡️ 4. Hallucination Verification</div>',
                    unsafe_allow_html=True,
                )

                st.caption(
                    "Each factual claim in the generated answer "
                    "is independently checked against evidence "
                    "retrieved from the uploaded PDF."
                )

                with st.spinner(
                    "Extracting claims and verifying evidence..."
                ):

                    verification_results = (
                        verify_answer(
                            answer=answer,
                            vector_store=(
                                st.session_state[
                                    "vector_store"
                                ]
                            ),
                            k=2,
                        )
                    )


                # ==========================================
                # NO CLAIMS
                # ==========================================

                if not verification_results:

                    st.info(
                        "No factual claims were available "
                        "for verification."
                    )


                # ==========================================
                # DISPLAY CLAIMS
                # ==========================================

                else:

                    for index, result in enumerate(
                        verification_results,
                        start=1,
                    ):

                        st.markdown(
                            f"### Claim {index}"
                        )

                        st.markdown(
                            f"""
                            <div class="claim-card">
                                <strong>
                                {result["claim"]}
                                </strong>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                        verdict = result[
                            "verdict"
                        ]

                        confidence = result[
                            "confidence"
                        ]


                        # ----------------------------------
                        # VERDICT
                        # ----------------------------------

                        if verdict == "SUPPORTED":

                            st.success(
                                f"✅ SUPPORTED  •  "
                                f"Confidence: "
                                f"{confidence:.2%}"
                            )

                        elif verdict == "CONTRADICTED":

                            st.error(
                                f"❌ CONTRADICTED  •  "
                                f"Confidence: "
                                f"{confidence:.2%}"
                            )

                        else:

                            st.warning(
                                f"⚠️ NO EVIDENCE  •  "
                                f"Confidence: "
                                f"{confidence:.2%}"
                            )


                        # ----------------------------------
                        # REASON
                        # ----------------------------------

                        if result.get(
                            "reason"
                        ):

                            st.write(
                                f"**Reason:** "
                                f"{result['reason']}"
                            )


                        # ----------------------------------
                        # EVIDENCE
                        # ----------------------------------

                        with st.expander(
                            f"📖 Evidence for Claim {index}"
                        ):

                            st.write(
                                result["evidence"]
                            )


                    # ==========================================
                    # GROUNDING SCORE
                    # ==========================================

                    grounding_result = (
                        calculate_grounding_score(
                            verification_results
                        )
                    )

                    st.divider()

                    st.markdown(
                        '<div class="section-title">📊 5. Overall Grounding</div>',
                        unsafe_allow_html=True,
                    )


                    # ----------------------------------
                    # SCORE
                    # ----------------------------------

                    score = grounding_result[
                        "grounding_score"
                    ]

                    verdict = grounding_result[
                        "overall_verdict"
                    ]

                    score_col, verdict_col = (
                        st.columns(2)
                    )


                    with score_col:

                        st.metric(
                            "Grounding Score",
                            f"{score:.2f}%",
                        )


                    with verdict_col:

                        st.metric(
                            "Overall Verdict",
                            verdict,
                        )


                    # ----------------------------------
                    # CLAIM COUNTS
                    # ----------------------------------

                    st.markdown(
                        "### Verification Summary"
                    )

                    col1, col2, col3 = (
                        st.columns(3)
                    )


                    with col1:

                        st.success(
                            f"✅ Supported\n\n"
                            f"**{grounding_result['supported']}**"
                        )


                    with col2:

                        st.warning(
                            f"⚠️ No Evidence\n\n"
                            f"**{grounding_result['no_evidence']}**"
                        )


                    with col3:

                        st.error(
                            f"❌ Contradicted\n\n"
                            f"**{grounding_result['contradicted']}**"
                        )


                    # ----------------------------------
                    # PROGRESS BAR
                    # ----------------------------------

                    st.markdown(
                        "### Grounding Level"
                    )

                    st.progress(
                        min(
                            max(
                                score / 100,
                                0.0,
                            ),
                            1.0,
                        )
                    )

                    st.caption(
                        "The grounding score represents the "
                        "proportion of generated claims that "
                        "were supported by retrieved evidence."
                    )


            except Exception as error:

                st.error(
                    f"An error occurred: {error}"
                )