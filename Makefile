.PHONY: dev dev-build rebuild down logs prod

dev:
	docker compose --env-file .env up

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
