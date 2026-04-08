---
title: Niva Prenatal Health
emoji: "🤰"
colorFrom: pink
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# Niva — AI Prenatal Health Monitor

Niva is an AI-powered prenatal health monitoring system designed to support pregnancy check-ins, risk assessment, and OpenEnv-compatible evaluation.

## What This Space Includes

- FastAPI backend with OpenEnv endpoints: `/reset`, `/step`, `/state`
- Prenatal diagnosis flow with user setup and check-ins
- Browser UI served from `preview.html`
- Docker-ready deployment for Hugging Face Spaces

## Tech Stack

- FastAPI
- SQLAlchemy
- Pydantic
- SQLite
- Rule-based maternal health risk engine

## Deployment

This Space runs as a Docker app on port `7860`.
