---
title: Churn Prediction API
emoji: 📊
colorFrom: blue
colorTo: purple
sdk: docker
app_file: Dockerfile
pinned: false
---

# Churn Prediction MLOps Pipeline 📊

Fully automated ML pipeline: DVC → MLflow → FastAPI → Docker → CI/CD

![CI/CD](https://github.com/aiankit/churn-prediction/actions/workflows/ci.yml/badge.svg)

## Live API
Your model is running at this Space's URL. Send a POST request to `/predict` with customer data to get a churn prediction.

## Local Test
docker pull aiankit/churn-api:latest
docker run -p 8000:8000 aiankit/churn-api:latest