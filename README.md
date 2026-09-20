# Multi-Stage Retrieval for BDI-II Symptom Ranking in Social Media (eRisk 2025)

## Authors
- Kulsoom Zaidi
- Alina Baig

## Project Overview
This project presents a multi-stage retrieval and ranking pipeline developed for **CLEF eRisk 2025 Task 1: Search for Symptoms of Depression**. 

The objective is to identify and rank sentences from a massive corpus of social media posts based on their relevance to specific depressive symptoms. The symptoms are defined by the **Beck Depression Inventory-II (BDI-II)**, a standard clinical questionnaire comprising 21 symptom items (e.g., Sadness, Loss of Energy, Sleep Changes, Suicidal Thoughts).

To tackle the challenge of lexical mismatch and semantic nuance in mental health text, we implemented a progressive 4-stage retrieval architecture, culminating in an LLM-based few-shot reranker.

---

## Task Description
**Task 1 — Search for Symptoms of Depression**  
Given a large text collection and 21 BDI-II symptom queries, the system must score and rank up to 1,000 sentences per symptom. Higher scores indicate stronger relevance to the user's *own* experience of the symptom.

---

## Methodology & Pipeline
The system employs a cascaded retrieval approach to balance computational efficiency with deep semantic understanding:

### 1. BM25 Baseline Retrieval
* **Mechanism**: Lexical matching using SQLite FTS5 (Full-Text Search 5) with Porter stemming.
* **Role**: Acts as the initial candidate generation stage over a corpus of **17.5M+ documents**, retrieving the top-1,000 candidates per BDI query.

### 2. Semantic Reranking (SBERT)
* **Mechanism**: Uses `all-MiniLM-L6-v2` to encode queries and BM25 candidates into dense vector embeddings.
* **Role**: Reranks the top-1,000 candidates based on cosine similarity, capturing semantic context that keyword matching misses.

### 3. Hybrid Scoring (BM25 + SBERT)
* **Mechanism**: Combines normalized BM25 lexical scores with SBERT semantic scores using a weighted formula (`0.2 * BM25 + 0.8 * SBERT`).
* **Role**: Leverages the exact-match strength of BM25 and the contextual awareness of neural embeddings.

### 4. LLM-Based Relevance Scoring (Mistral-7B)
* **Mechanism**: Uses a 4-bit quantized **Mistral-7B-Instruct** model.
* **Prompting**: Employs hardcoded few-shot examples for each of the 21 BDI items to guide the model in distinguishing between *personal symptom expression* vs. *general/hypothetical mentions*.
* **Scoring**: Instead of generative text classification, it extracts **continuous relevance probabilities (0.0 to 1.0)** by computing the softmax of the logits for the target tokens (`relevant` vs `not`) at the final sequence position.

---

## Evaluation & Results
The systems were evaluated using the official TREC evaluation tool (`trec_eval`) against the provided ground truth (`qrels`). 

| Retrieval System | MAP | nDCG@10 | P@10 |
| :--- | :---: | :---: | :---: |
| **BM25 Baseline** | 0.0515 | 0.6240 | 0.5905 |
| **Rerank MiniLM** | 0.0537 | 0.6384 | 0.6238 |
| **Hybrid MiniLM** | 0.0535 | 0.6394 | 0.6238 |
| **Mistral LLM (Few-Shot)** | **0.0572** | **0.7097** | **0.6905** |

***Conclusion**: The LLM-based few-shot reranker significantly outperforms traditional lexical and embedding-based methods, particularly in nDCG@10 and Precision@10, proving its ability to deeply understand clinical context in noisy social media text.*

---

## Tech Stack & Dependencies
* **Language**: Python 3.10+
* **Deep Learning**: PyTorch, Hugging Face `transformers`, `sentence-transformers`
* **LLM Quantization**: `bitsandbytes` (4-bit NF4 quantization for Mistral-7B)
* **Database / Indexing**: SQLite3 (FTS5)
* **Evaluation**: `trec_eval`
* **Environment**: Google Colab (T4 GPU)

### Installation
```bash
pip install torch transformers accelerate evaluate bitsandbytes sentence-transformers faiss-cpu tqdm
