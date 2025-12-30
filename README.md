# Real Estate RAG & VLM Evaluation System

This repository contains the implementation for the "LLM for Real-Estate Semantic Retrieval & Recommendation" assignment. It includes a RAG system for precise property retrieval and a comprehensive evaluation framework for VLM-generated tags.

## 📂 Project Structure

- `data/`: Contains dataset files (raw and processed).
- `services/`: Core logic for RAG and VLM services.
- `infrastructures/`: Database clients (Neo4j) and utilities.
- `scripts/`: Executable scripts for data processing, tasks, and evaluation.
  - `property_search_recommendation/`: Scripts for Task 1 (RAG).
  - `vlm_tag_quality_service/`: Scripts for Task 2 (VLM Eval).
- `docker-compose.yml`: Neo4j database configuration.

## 🚀 Setup & Installation

### 1. Prerequisites
- Python 3.12+
- Docker & Docker Compose
- [uv](https://github.com/astral-sh/uv) (recommended) or pip

### 2. Environment Configuration
Copy the sample environment file and configure your API keys.

```bash
cp .env.sample .env
```

Edit `.env` and provide:
- `OPENAI_API_KEY`
- `GEMINI_AI_STUDIO_API_KEY`
- Neo4j credentials (default: neo4j/password)

### 3. Start Database
Start the Neo4j graph database using Docker.

```bash
docker-compose up -d
```

### 4. Install Dependencies
Using `uv`:
```bash
uv sync
```
Or using `pip`:
```bash
pip install -r pyproject.toml
```

---

## 📌 Task 1: RAG System for Property Retrieval

This task implements a hybrid search system (Vector + Knowledge Graph) to retrieve properties based on user queries.

### 1. Data Preparation
Clean the raw data and prepare it for import.

```bash
cd scripts/property_search_recommendation
export PYTHONPATH=../../
python clean_raw_data.py
```

### 2. Import to Neo4j
Import the cleaned data into the Neo4j graph database.

```bash
python import_properties.py
```

### 3. Generate Embeddings
Generate vector embeddings for property descriptions using OpenAI.

```bash
python embed_properties_openai.py
```

### 4. Run End-to-End Evaluation
Run the retrieval evaluation against the generated testing dataset.

```bash
# Optional: Generate new test questions
# python generate_testing_dataset.py

python task_1_end_2_end_test.py
```

**Architecture:**
- **Query Understanding:** LLM extracts "Hard Filters" (Cypher) and "Soft Filters" (Vector Search) from the user query.
- **Retriever:** Hybrid approach combining Neo4j Cypher queries (for precise constraints like location, price) and Vector Similarity Search (for semantic matching).

---

## 📌 Task 2: Evaluation Framework for VLM Tags

This task evaluates the quality of AI-generated tags using a "VLM-as-a-Judge" approach and Spatial Signal verification.

### 1. Data Rematching (Pre-processing)
Fix alignment issues between images and text descriptions.

```bash
cd scripts/vlm_tag_quality_service
export PYTHONPATH=../../
python vlm_tag_data_rematching.py
```

### 2. Add Raw Descriptions
Inject raw descriptions back into the dataset for context.

```bash
python add_raw_description.py
```

### 3. Run VLM-as-a-Judge Evaluation
Evaluate tag correctness, specificity, and redundancy using Gemini.

```bash
python vlm_as_a_judge_evalution.py
```

### 4. Run Spatial Signal Evaluation
Evaluate spatial accuracy using Depth-Anything-V2 depth maps.

```bash
python vlm_with_spatial_signals_as_a_judge_evaluation.py
```

**Metrics:**
- **Confidence Score (0-5):** Overall quality rating.
- **Spatial Accuracy:** Verification against depth maps to detect hallucinations.

---

## 🛠️ Technologies Used

- **LLMs:** OpenAI GPT-4o, Google Gemini 1.5 Pro/Flash
- **Database:** Neo4j (Graph DB)
- **Vision:** CLIP, Depth-Anything-V2
- **Frameworks:** LangChain, Pydantic
