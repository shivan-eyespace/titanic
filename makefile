.PHONY: run
run:
	@echo "Running containers..." && \
	docker-compose down --remove-orphans && \
	docker-compose up --build

.PHONY: run-d
run-d:
	@echo "Running containers in detached mode..." && \
	docker-compose down --remove-orphans && \
	docker-compose up --build -d

.PHONY: migrate
migrate:
	@echo "Migrating" && \
	docker-compose up db -d && \
	docker-compose run migrate -e "DATABASE_URL" up

.PHONY: migrate-down
migrate-down:
	@echo "Migrating" && \
	docker-compose up db -d && \
	docker-compose run migrate -e "DATABASE_URL" down

.PHONY: migration
migration:
	@echo "Creating new migration" && \
	docker-compose up db -d && \
	docker-compose run migrate -e "DATABASE_URL" new $(NEW)

ARGPATH="test"
.PHONY: api-tests
api-tests:
	@echo "Running API tests" && \
	docker-compose up -d && \
	docker-compose exec api sh -c "pytest -vv -k $(ARGPATH) --cov-report html --cov='src'"

.PHONY: connect-db
connect-db:
	@echo "Connecting to database" && \
	docker-compose up db -d && \
	docker-compose exec db sh -c "psql -U postgres"
