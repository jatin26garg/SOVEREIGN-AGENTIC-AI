# 🛡️ Sovereign AI Workbench

**A self-hosted, air-gapped AI workbench for defense, PSU, and government environments.**

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)
[![Qdrant](https://img.shields.io/badge/Qdrant-1.9+-red.svg)](https://qdrant.tech/)
[![BGE-M3](https://img.shields.io/badge/BGE--M3-1.0-purple.svg)](https://huggingface.co/BAAI/bge-m3)
[![Qwen3](https://img.shields.io/badge/Qwen3--8B-1.0-orange.svg)](https://ollama.com/library/qwen3:8b)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📋 Table of Contents

- [Overview](#-overview)
- [The Problem](#-the-problem)
- [Architecture](#-architecture)
- [Features](#-features)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Configuration](#-configuration)
- [API Reference](#-api-reference)
- [How It Works](#-how-it-works)
- [Workspace Layout](#-workspace-layout)
- [Testing](#-testing)
- [Roadmap](#-roadmap)
- [SIH Alignment](#-sih-alignment)
- [License](#-license)

---

## 🎯 Overview

**Sovereign AI Workbench** is a fully local, air-gapped AI assistant that runs entirely on an
organization's own infrastructure. It provides a chat + agent interface over internal documents,
performs hybrid search, and coordinates tools to produce real deliverables — with **zero external
API calls**.

Built for environments where cloud AI assistants (ChatGPT, Claude, Codex) are prohibited due to
data sovereignty, this system gives industrial users a Claude/Codex-style workbench they can
actually use — without any data leaving the premises.

**Core capabilities:**

- 📄 **Document RAG** — Upload PDFs, DOCX, TXT; ask questions; get cited answers
- 🔍 **Hybrid Search** — Combines semantic + keyword retrieval via BGE-M3 + Qdrant
- 🤖 **Agent Orchestrator** — Plans and executes multi-step tasks using tools
- 📁 **File System Tool** — Sandboxed read/write within a workspace
- 🔒 **Zero External Calls** — Verified via network monitoring

---

## ❗ The Problem

Refineries, PSUs, defense-linked manufacturing, and government offices generate enormous volumes
of sensitive knowledge work — approval notes, board presentations, engineering calculations,
internal code, scanned drawings, inspection reports. None of this can go through cloud AI because
the underlying data is confidential (P&IDs, financials, vendor negotiations, unreleased designs).

Company policy keeps this data on-premises, so today people either:

1. Do the work **manually** → lost productivity, or
2. **Quietly paste** confidential material into public tools → data leakage

**Nothing deployable exists today** that industrial users can work with the way they use
Claude or Codex — until now.

---

## AGENT GRAPH
init → router ─┬─ end (greeting/vague) ────────────────────────────────→ DONE
                └─ plan → execute_step ─┬─ ok, more steps ──→ execute_step (loop)
                                        ├─ ok, done ────────→ finalize
                                        └─ fail/empty ──────→ replan ─┬─→ execute_step
                                                                      └─→ finalize
                                                                            │
                                                            finalize → critique ─┬─ retry ─→ plan
                                                                                 └─ done ──→ DONE
## 🏗️ Architecture

┌─────────────────────────────────────────────────────────────────────────────┐
│ USER INTERFACE │
│ (Next.js / API Clients) │
└─────────────────────────────────┬───────────────────────────────────────────┘
│ HTTP/REST
▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ FASTAPI BACKEND (Local) │
│ │
│ ┌─────────────────────────────────────────────────────────────────────┐ │
│ │ AGENT ORCHESTRATOR (LangGraph) │ │
│ │ │ │
│ │ INIT → PLAN → EXECUTE_STEP ──┐ │ │
│ │ ▲ │ (loop until done) │ │
│ │ └────────┘ │ │
│ │ │ │ │
│ │ ▼ │ │
│ │ FINALIZE │ │
│ └─────────────────────────────────────────────────────────────────────┘ │
│ │ │
│ ▼ │
│ ┌─────────────────────────────────────────────────────────────────────┐ │
│ │ TOOLS │ │
│ │ ┌─────────────────┐ ┌─────────────────────────────────────────┐ │ │
│ │ │ RAG Tool │ │ File System Tool │ │ │
│ │ │ • Hybrid search│ │ • Sandboxed read/write │ │ │
│ │ │ • Citations │ │ • Path traversal protection │ │ │
│ │ │ • Qdrant + BM25│ │ • Extension & size limits │ │ │
│ │ └─────────────────┘ └─────────────────────────────────────────┘ │ │
│ └─────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────┬───────────────────────────────────────────┘
│
┌─────────────────────────┼─────────────────────────┐
│ │ │
▼ ▼ ▼
┌──────────────────┐ ┌──────────────────────┐ ┌──────────────────────┐
│ BGE-M3 │ │ Qdrant │ │ Qwen3-8B │
│ (Local Model) │ │ (Vector DB) │ │ (via Ollama) │
│ │ │ │ │ │
│ • Dense (1024d) │ │ • Dense vectors │ │ • Local inference │
│ • Sparse (BM25) │ │ • Sparse vectors │ │ • No external calls │
│ • 100+ langs │ │ • RRF hybrid search │ │ • 8K context │
└──────────────────┘ └──────────────────────┘ └──────────────────────┘


**Every component runs locally. No data leaves the machine.**

---

## ✨ Features

### Core Capabilities

| Feature | Description |
|---------|-------------|
| **Hybrid Search** | Dense (semantic) + sparse (keyword) retrieval with RRF fusion |
| **Multi-Format Ingestion** | PDF, DOCX, TXT document support |
| **Agentic Workflows** | LLM-driven planning + multi-step execution |
| **Sandboxed File Tools** | Path-traversal-safe file I/O in a dedicated workspace |
| **Source Citations** | Every answer includes traceable source references |
| **Air-Gapped** | Zero external network calls — verifiable |

### Technical Highlights

| Component | Choice | Why |
|-----------|--------|-----|
| **Embeddings** | BAAI/bge-m3 | Dense + sparse in one pass; 100+ languages; 8192-token context |
| **Vector DB** | Qdrant | Native hybrid search; production-ready; no cloud dependency |
| **LLM** | Qwen3-8B (Ollama) | Fully open-weight; runs locally; no API key required |
| **Backend** | FastAPI | Fast async; auto OpenAPI docs; strong typing |
| **Agent Framework** | LangGraph | State-machine agent with explicit nodes/edges |
| **Document Parsing** | pypdf, python-docx | Robust text extraction |

---

## 🧰 Tech Stack

### Backend Dependencies

```txt
# Web Framework
fastapi==0.104.1
uvicorn[standard]==0.24.0
python-multipart==0.0.6

# Vector Database
qdrant-client==1.9.0

# Embeddings
FlagEmbedding==1.2.0

# LLM Integration
langchain==0.1.0
langchain-ollama==0.1.0
langgraph==0.0.40

# Document Parsing
pypdf==3.17.0
python-docx==1.1.0

# Utilities
python-dotenv==1.0.0
pydantic==2.5.0

# Testing
pytest==7.4.0
pytest-asyncio==0.21.0


rag-project/
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py                     # FastAPI app & routes
│   │   ├── config.py                   # Settings & env vars
│   │   ├── models.py                   # Pydantic schemas
│   │   │
│   │   ├── services/
│   │   │   ├── __init__.py
│   │   │   └── rag_service.py          # Core RAG (hybrid search + generation)
│   │   │
│   │   ├── tools/
│   │   │   ├── __init__.py
│   │   │   └── file_tools.py           # Sandboxed file I/O
│   │   │
│   │   ├── agent/                      # Agent Orchestrator
│   │   │   ├── __init__.py
│   │   │   ├── orchestrator.py         # LangGraph workflow + run_agent()
│   │   │   ├── nodes.py                # init / plan / execute_step / finalize
│   │   │   ├── state.py                # AgentState TypedDict
│   │   │   └── tools.py                # Tool wrappers (rag_search, write_output)
│   │   │
│   │   └── utils/
│   │       ├── __init__.py
│   │       ├── file_handlers.py        # PDF/DOCX/TXT extraction
│   │       └── chunking.py             # Text chunking with overlap
│   │
│   ├── tests/
│   │   ├── test_rag_service.py
│   │   ├── test_file_tools.py
│   │   └── test_agent.py
│   │
│   ├── requirements.txt
│   ├── .env.example
│   └── .env                            # (git ignored)
│
├── workspace/                          # Agent sandbox (git ignored)
│   ├── inputs/                         # Uploaded documents
│   ├── outputs/                        # Agent-generated files
│   └── temp/                           # Scratch space
│
├── docker-compose.yml                  # Qdrant + Ollama + Backend
├── README.md
└── LICENSE


LICENSE
🚀 Installation
Prerequisites
Python 3.10+

Docker (for Qdrant + Ollama)

8 GB+ RAM (16 GB recommended)

~10 GB disk (for models)


Step 1: Clone

git clone https://github.com/yourusername/rag-project.git
cd rag-project/backend


