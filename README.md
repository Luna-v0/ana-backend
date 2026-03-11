# Attorney Normative Assistent (ANA)

**Application for attorneys to assist with legal tasks**

Version 1.0 — February 2026

This projects is produced in portuguese since it is intended for brazilian attorneys.

## Executive Summary

This system is a complete legal assistance platform that runs **mostly locally**, ensuring the privacy of case data. It combines:

- **Hybrid RAG** (semantic + syntactic) over all Brazilian legislation
- **Intelligent transcription** of hearings with participant identification
- **Generation and editing** of Word documents with legal formatting
- **Isolated sessions** per case with dedicated RAG
- **Automatic validation** of cited laws to minimize hallucinations
- **Agent architecture** modular orchestrated via LangGraph

The main stack is: **Ollama** (LLM), **Qdrant** (vector DB), **WhisperX** (transcription), **LangGraph** (agents), **FastAPI** (backend).

For the documentation, see the [docs](docs/) folder. It is in portuguese.