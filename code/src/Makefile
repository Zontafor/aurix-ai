# Makefile for AURIX-AI
PY=python
API=src.api.main:app

.PHONY: help train evaluate api docker-up docker-down clean

help:
	@echo "Targets:"
	@echo "  train       - Run XGBoost training locally"
	@echo "  evaluate    - Run evaluation summary"
	@echo "  api         - Start FastAPI locally (uvicorn)"
	@echo "  docker-up   - Build and start services via docker-compose"
	@echo "  docker-down - Stop and remove compose services"
	@echo "  clean       - Remove Python artifacts and generated files"

train:
	$(PY) aurix_model_train_xgb.py

evaluate:
	$(PY) aurix_evaluate.py

api:
	uvicorn $(API) --host 0.0.0.0 --port 8080 --reload

docker-up:
	docker compose up --build -d

docker-down:
	docker compose down

clean:
	rm -rf __pycache__ .pytest_cache .mypy_cache mlruns *.joblib *.npy *.json *.pdf *.csv
