.PHONY: install run test lint format check docker-build docker-up docker-down docker-logs

install:        ## Create the virtualenv with dev tools
	uv sync

run:            ## Run the API locally with auto-reload on http://localhost:8000 (docs at /docs)
	uv run uvicorn firebridge_agent.main:create_app --factory --reload --port 8000

test:           ## Run the test suite (no network)
	uv run pytest -q

lint:           ## Static checks
	uv run ruff check src tests
	uv run ruff format --check src tests

format:         ## Auto-format
	uv run ruff check --fix src tests
	uv run ruff format src tests

check: lint test

docker-build:   ## Build the production image
	docker compose build

docker-up:      ## Start the service in the background (reads .env)
	docker compose up -d

docker-down:
	docker compose down

docker-logs:
	docker compose logs -f agent
