# Timbre: Search Samples the Way You Hear Them

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![MongoDB](https://img.shields.io/badge/MongoDB-47A248?style=for-the-badge&logo=mongodb&logoColor=white)
![Qdrant](https://img.shields.io/badge/Qdrant-DC244C?style=for-the-badge&logo=qdrant&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![OpenAI](https://img.shields.io/badge/OpenAI-412991?style=for-the-badge)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)

**Timbre is a sample-search assistant for music producers.** Sample libraries describe sounds with technical tags and file names, but producers think in words like "punchy", "warm" or "dark". Timbre lets you find samples the way you hear them:

* **Describe a sound in plain words** ("a distorted synth bass that feels very dark") and get matching samples, plus effect settings to get closer to that sound.
* **Drop in an audio file** and get a description of how it sounds, clean tags, and the most similar samples in the library.

Under the hood it combines a sample library enriched with LLM-written captions, CLAP audio-text embeddings and a team of AI agents (agentic RAG). It is served through a FastAPI chat endpoint; the screenshots below show it as a plugin inside a DAW.

## Text-to-audio search
<img width="800" alt="Text-to-audio search" src="docs/images/plugin_text_search.png" />

## Audio-to-audio search
<img width="800" alt="Audio analysis" src="docs/images/plugin_audio_analysis.png" />

## Overview

In music production, there is a "semantic gap" between technical, machine-readable metadata and the subjective, perceptual language (e.g., "punchy," "warm") humans use to describe sound. This project bridges that gap using a high-precision, semantically enriched dataset, Large Language Models (LLMs), and Contrastive Language-Audio Pretraining (CLAP) to create an intelligent, Audio-Native Agentic RAG assistant.

<img width="800" alt="From signal to meaning: the semantic gap" src="docs/images/semantic_gap.png" />

---

## Architecture

The project is split into four phases, each living in its own folder under `src/`. The first three run once to build the dataset; the last one answers every request.

<img width="800" alt="Project phases: data extraction, data exploration, data ingestion and agentic RAG" src="docs/images/project_phases.png" />

| Phase | Folder | Services | Output |
| --- | --- | --- | --- |
| 1. Data extraction | `src/data_retrievial/` | <img src="https://cdn.simpleicons.org/python/3776AB" width="18" alt="Python"/> <img src="https://cdn.simpleicons.org/mongodb/47A248" width="18" alt="MongoDB"/> | Audio in GridFS, metadata in MongoDB |
| 2. Data exploration | `src/data_exploration/` | <img src="https://cdn.simpleicons.org/python/3776AB" width="18" alt="Python"/> <img src="https://cdn.simpleicons.org/mongodb/47A248" width="18" alt="MongoDB"/> | Descriptor vocabulary |
| 3. Data ingestion | `src/data_ingestion/` | <img src="https://cdn.simpleicons.org/python/3776AB" width="18" alt="Python"/> <img src="https://cdn.simpleicons.org/mongodb/47A248" width="18" alt="MongoDB"/> <img src="https://cdn.simpleicons.org/qdrant/DC244C" width="18" alt="Qdrant"/> | Qdrant collections ready to search |
| 4. Agentic RAG | `src/rag/`, `src/api.py` | <img src="https://cdn.simpleicons.org/python/3776AB" width="18" alt="Python"/> <img src="https://cdn.simpleicons.org/qdrant/DC244C" width="18" alt="Qdrant"/> <img src="https://cdn.simpleicons.org/fastapi/009688" width="18" alt="FastAPI"/> | Samples, labels and DSP recipes |

Shared code lives in `src/core/` (domain models, MongoDB/GridFS repositories and the Qdrant repository) and `src/config/settings.py`.

### 1. Data extraction · `src/data_retrievial/`

Collects the raw material and stores it in **MongoDB**, with the audio binaries split into **GridFS** chunks.

* **SampleFocus extractor** · `src/data_retrievial/sample_focus/SampleFocusExtractor.py`
  Scrapes audio samples and their tags from [SampleFocus](https://samplefocus.com). `sample_focus/main.py` drives it with a *Matrix Selection Strategy*: 11 instruments (bass, drums and synth families) crossed with 8 timbres (warm, cold, soft, happy, heavy, airy, bright, dark), up to 250 samples per cell, skipping physically implausible pairs such as an airy sub bass. `metadata.py` parses the sample page and `privacy_utils.py` rate-limits the requests.
* **SocialFX extractor** · `src/data_retrievial/socialfx/socialfx_extractor.py`
  Loads the [SocialFX](https://huggingface.co/datasets/seungheondoh/socialfx-original) dataset from Hugging Face and stores a knowledge base that maps perceptual descriptors (e.g. "warm") to real EQ, compression and reverb parameters.

### 2. Data exploration · `src/data_exploration/`

* `extract_descriptors.py` counts how often each SocialFX descriptor is used and keeps the ones with at least 8 occurrences, saved to `descriptors_list.txt`.

### 3. Data ingestion · `src/data_ingestion/`

Turns the stored samples into searchable vectors in **Qdrant**. Entry point: `src/data_ingestion/main.py`.

<img width="800" alt="Semantic enrichment pipeline" src="docs/images/enrichment_pipeline.png" />

* **Semantic enrichment** · `ingestors/enrich_audio_doublevectors.py`
  For every sample in GridFS, the LabelEnricher tool (GPT-4o) rewrites the noisy user tags into a natural-language caption and a cleaner tag set.
* **Hallucination check** · `src/rag/tools/audio_analysis.py`
  CLAP embeds both the caption and the audio. If their cosine similarity is at least 0.25 the caption is marked *Verified*, otherwise *Low confidence*; the score is stored as `clap_score` so answers can show how much to trust each label.
* **Dual-vector indexing**
  Each sample becomes one point in the `audio_enriched` collection with two named vectors: `text_vector` (caption, `text-embedding-3-small`, 1536-d) and `audio_vector` (CLAP HTSAT, 512-d).
* **DSP parameters** · `ingestors/ingest_parameters.py`
  Embeds the SocialFX descriptors into their own `socialfx_vectors` collection.

<img width="300" height="300" alt="audio_retrieval" src="https://github.com/user-attachments/assets/0847735d-83bf-4280-bdb7-451a5bd899a4" />

### 4. Agentic RAG · `src/rag/` and `src/api.py`

A **FastAPI** `/chat` endpoint (`src/api.py`) runs the agent workflow in `src/rag/workflow.py`.

<img width="800" alt="Agentic orchestration" src="docs/images/agentic_orchestration.png" />

* **Intent Classifier** · decides whether a request is a retrieval (text) or an analysis (audio).
* **Retrieval** · the Audio Retriever ranks samples by caption similarity on `text_vector`, while the Sound Designer looks up DSP parameters for the adjectives in the query in `socialfx_vectors`. Both run in parallel.
* **Analysis (Reverse RAG)** · the Audio Analyst fingerprints the uploaded audio with CLAP, finds its nearest neighbours on `audio_vector`, and the Label Enricher writes a label from their tags.
* **Humanizer** · turns the JSON results into the final answer, with audio previews.

Agents live in `src/rag/agents/`, their prompts in `src/rag/prompts/` and the CLAP model wrapper in `src/rag/clap/`.

---

## Getting started

1. Start MongoDB and Qdrant:
   ```bash
   docker compose up -d
   ```
2. Install the dependencies (PyTorch with CUDA is listed in `requirements.txt`):
   ```bash
   pip install -r requirements.txt
   ```
3. Set `OPENAI_API_KEY` in your `.env`.
4. Build the dataset and run the workflow: `PYTHONPATH=src python main.py` runs the SampleFocus and SocialFX extraction, the ingestion and the RAG workflow in sequence.
5. Serve the API:
   ```bash
   cd src && python api.py
   ```

---

## Key Results

* **Text-to-Audio Semantic Search:** Our Ablation Study proved that substituting raw, noisy tags with LLM-synthesized captions significantly improved retrieval ranking quality, increasing the **nDCG@5 metric from 0.5851 to 0.8460**.
* **Audio-to-Audio Search:** The CLAP-powered analysis successfully bypasses the human "vocabulary mismatch" problem, relying strictly on acoustic features to find highly coherent topological neighbors even when human-assigned tags completely diverge.

## IMPORTANT NOTE
The front-end is yet to be implemented (we used a mockup)
