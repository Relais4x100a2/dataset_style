.PHONY: dev dev-build rebuild down logs prod

dev:
	docker compose --env-file .env up
	# Lance PostgreSQL, SuperTokens, Streamlit (:8501) et le slice web FastAPI (:8080 par défaut).

dev-build:
	docker compose --env-file .env up --build

rebuild:
	docker compose --env-file .env build --no-cache

down:
	docker compose --env-file .env down -v

logs:
	docker compose --env-file .env logs -f app

prod:
	caprover deploy
