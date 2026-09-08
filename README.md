# 🛡️ HallucinationGuard

### RAG-Based Hallucination Detection and Claim Verification System

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://hallucinationguard-f6dyaeftfdwafkjrur9wmd.streamlit.app/)

HallucinationGuard is an end-to-end **Retrieval-Augmented Generation (RAG)** system that answers questions from uploaded documents and verifies whether the generated answer is actually supported by the source document.

Unlike a basic RAG application that only retrieves context and generates an answer, HallucinationGuard adds a **claim-level verification layer** to detect unsupported or potentially hallucinated statements.

---

## 🚀 Live Demo

👉 **[Try HallucinationGuard](https://hallucinationguard-f6dyaeftfdwafkjrur9wmd.streamlit.app/)**

Upload a PDF, ask questions about its contents, and inspect how each factual claim in the generated answer is verified against the document.

---

## ✨ Features

- 📄 Upload a PDF document
- ✂️ Automatically split documents into smaller chunks
- 🔎 Semantic search using vector embeddings
- 🗄️ Session-specific ChromaDB vector store
- 🤖 Gemini-powered RAG question answering
- 🧩 Automatic factual claim extraction
- 🔍 Independent evidence retrieval for each claim
- ✅ Claim-level verification
- ⚠️ Detect contradicted claims
- ❓ Identify claims with no supporting evidence
- 📊 Confidence scores for individual claims
- 📈 Overall grounding score
- 🎯 Overall grounding verdict
- 🌐 Deployed using Streamlit Community Cloud

---

## 🧠 How It Works

HallucinationGuard follows a multi-stage verification pipeline:

```text
                    ┌─────────────────────┐
                    │     Upload PDF      │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │  Document Loading   │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │      Chunking       │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │     Embeddings      │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │     ChromaDB        │
                    │    Vector Store     │
                    └──────────┬──────────┘
                               │
                         User Question
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Semantic Retrieval  │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │  Gemini RAG Answer  │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │  Claim Extraction  │
                    └──────────┬──────────┘
                               │
                     Individual Claims
                               │
                               ▼
                    ┌─────────────────────┐
                    │  Evidence Retrieval │
                    │      Per Claim      │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Gemini Verification │
                    └──────────┬──────────┘
                               │
                ┌──────────────┼──────────────┐
                ▼              ▼              ▼
           SUPPORTED      CONTRADICTED    NO_EVIDENCE
                │              │              │
                └──────────────┼──────────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Grounding Score   │
                    │   & Overall Verdict │
                    └─────────────────────┘
```

---

## 🔍 Why Claim-Level Verification?

A conventional RAG system generally follows:

```text
Question → Retrieve Context → Generate Answer
```

Even when relevant context is retrieved, the generated response may contain unsupported statements or information that is not actually present in the source document.

HallucinationGuard therefore adds an additional verification layer:

```text
Generated Answer
       ↓
Extract Individual Claims
       ↓
Retrieve Evidence for Each Claim
       ↓
Verify Claims Independently
       ↓
Calculate Grounding Score
```

This allows the system to evaluate the generated response at the **claim level** rather than treating the entire answer as simply correct or incorrect.

---

## ⚙️ Verification Logic

Each extracted claim is assigned one of three verdicts:

### ✅ SUPPORTED

The retrieved evidence directly states or clearly entails the claim.

### ❌ CONTRADICTED

The retrieved evidence explicitly conflicts with the claim.

### ❓ NO_EVIDENCE

The available evidence does not clearly support or contradict the claim.

Each claim also receives a **confidence score between 0 and 1** and a brief explanation based on the retrieved evidence.

---

## 📊 Grounding Score

HallucinationGuard calculates a confidence-weighted grounding score based on the claim verification results.

Supported claims contribute their verification confidence to the score.

Contradicted and unsupported claims contribute zero.

```text
Grounding Score =
(sum of confidence of supported claims / total claims) × 100
```

The overall result is classified as:

| Score | Overall Verdict |
|---|---|
| ≥ 80% | 🟢 HIGHLY GROUNDED |
| ≥ 50% | 🟡 PARTIALLY GROUNDED |
| < 50% | 🔴 POORLY GROUNDED |

> **Note:** The grounding score measures how well the generated response is supported by the retrieved evidence from the uploaded document. It does not measure absolute truth in the real world.

---

## 🏗️ Project Structure

```text
HallucinationGuard/
│
├── app/
│   ├── embeddings/
│   │   └── embedding_service.py
│   │
│   ├── generation/
│   │   └── generator.py
│   │
│   ├── ingestion/
│   │   └── loader.py
│   │
│   ├── processing/
│   │   └── chunker.py
│   │
│   ├── ui/
│   │   ├── pdf_processor.py
│   │   ├── rag_pipeline.py
│   │   ├── vector_store.py
│   │   └── verification_pipeline.py
│   │
│   └── verification/
│       ├── claim_extractor.py
│       └── scoring.py
│
├── .gitignore
├── requirements.txt
├── streamlit_app.py
└── README.md
```

---

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| **Python** | Core programming language |
| **LangChain** | RAG pipeline and document processing |
| **Gemini** | Answer generation, claim extraction, and verification |
| **Hugging Face** | Text embeddings |
| **ChromaDB** | Vector database |
| **Streamlit** | Web application interface |
| **PyPDF** | PDF document loading |
| **python-docx** | DOCX document support in backend |

---

## 🔄 Detailed Pipeline

### 1. Document Ingestion

The uploaded PDF is temporarily stored and loaded using a LangChain PDF loader.

### 2. Chunking

The document is divided into smaller overlapping chunks using `RecursiveCharacterTextSplitter`.

Current configuration:

```text
Chunk size:    500 characters
Chunk overlap: 100 characters
```

### 3. Embeddings

Each document chunk is converted into a numerical vector representation using Hugging Face embeddings.

### 4. Vector Storage

The embeddings are stored in a unique ChromaDB collection created for the uploaded document.

This provides a separate vector store for each uploaded PDF session.

### 5. Semantic Retrieval

When the user asks a question, the system retrieves the most semantically similar document chunks.

### 6. RAG Answer Generation

The retrieved context is provided to Gemini with instructions to answer using only the uploaded document.

If sufficient information cannot be found, the system responds:

```text
I don't have enough information in the uploaded PDF.
```

### 7. Claim Extraction

The generated answer is passed to Gemini to decompose the response into independently verifiable factual claims.

### 8. Evidence Retrieval

Each extracted claim is independently used as a retrieval query against ChromaDB to find supporting or contradicting evidence.

### 9. Claim Verification

Gemini evaluates each claim against its retrieved evidence and assigns one of:

```text
SUPPORTED
CONTRADICTED
NO_EVIDENCE
```

A confidence score and explanation are also generated for each claim.

### 10. Grounding Score

The individual verification results are aggregated into an overall grounding score and verdict.

---

## 🖥️ Application Flow

The Streamlit interface follows this workflow:

```text
Upload PDF
    ↓
Process Document
    ↓
Ask Question
    ↓
Generate Answer
    ↓
Retrieve Evidence
    ↓
Extract Claims
    ↓
Verify Claims
    ↓
Display Claim-Level Results
    ↓
Calculate Grounding Score
```

For each verified claim, the application displays:

- Claim
- Verdict
- Confidence
- Reason
- Retrieved evidence

---

## 🔐 API Key Configuration

HallucinationGuard uses a Gemini API key for its LLM operations.

### Local Development

Create a `.env` file in the project root:

```env
GOOGLE_API_KEY="your_api_key_here"
```

The `.env` file should **never be committed to GitHub**.

### Streamlit Cloud

For deployment, add the API key through Streamlit's **Secrets** configuration:

```toml
GOOGLE_API_KEY = "your_api_key_here"
```

---

## ▶️ Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/nissankararaoThanvitha/HallucinationGuard.git
```

### 2. Navigate into the project

```bash
cd HallucinationGuard
```

### 3. Create a virtual environment

```bash
python -m venv venv
```

### 4. Activate the environment

#### Windows PowerShell

```powershell
venv\Scripts\Activate.ps1
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

### 6. Configure the Gemini API key

Create a `.env` file:

```env
GOOGLE_API_KEY="your_api_key_here"
```

### 7. Start the application

```bash
streamlit run streamlit_app.py
```

The application will open in your browser.

---

## ☁️ Deployment

HallucinationGuard is deployed using **Streamlit Community Cloud**.

### 🌐 Live Application

**[Launch HallucinationGuard](https://hallucinationguard-f6dyaeftfdwafkjrur9wmd.streamlit.app/)**

The application connects to the GitHub repository and uses Streamlit Cloud for deployment.

---

## 🎯 Project Objective

The primary objective of HallucinationGuard is to improve the transparency and reliability of RAG-based applications.

Instead of only asking:

> **What did the model answer?**

HallucinationGuard also asks:

> **Which claims in the generated answer are actually supported by the source document?**

This makes it possible to inspect the grounding of individual factual claims rather than relying solely on the generated answer.

---

## 🚧 Current Limitations

- The Streamlit interface currently supports PDF uploads.
- Semantic retrieval always returns the configured number of closest chunks, even if the question may not be answerable from the document.
- Verification quality depends on the quality of retrieved evidence.
- The grounding score measures document grounding rather than absolute real-world truth.
- Gemini API latency and availability can affect response time.

---

## 🔮 Future Improvements

Potential improvements include:

- Retrieval relevance thresholds
- Hybrid keyword and semantic retrieval
- Improved evidence ranking
- Source citation and page-level highlighting
- Multi-document support
- Better structured LLM outputs
- Automated evaluation benchmarks
- Improved caching and latency
- Additional document formats in the Streamlit interface

---

## 👩‍💻 Author

### Thanvitha Nissankararao

GitHub:  
**[nissankararaoThanvitha](https://github.com/nissankararaoThanvitha)**

---

## ⭐ Support

If you find this project useful, consider giving the repository a ⭐ on GitHub.
