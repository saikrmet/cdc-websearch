#!/bin/bash
gunicorn -k uvicorn.workers.UvicornWorker tweets_analysis_app.main:app --bind=0.0.0.0:$PORT