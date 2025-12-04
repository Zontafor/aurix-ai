# src/api/main.py
# AURIX-AI FastAPI entrypoint.

from fastapi import FastAPI

# Relative imports within the src.api package
from .train import router as train_router
from .recommend import router as recommend_router
from .predict import router as predict_router
from explain import router as explain_router
from .routes.monitor import router as monitor_router

APP_NAME = "AURIX-AI API"
app = FastAPI(title=APP_NAME)

# Health check endpoint
@app.get("/health")
def health():
    """Lightweight health check endpoint."""
    return {"status": "ok"}

# Attach routers
app.include_router(train_router)
app.include_router(recommend_router)
app.include_router(predict_router)
app.include_router(explain_router)
app.include_router(monitor_router)