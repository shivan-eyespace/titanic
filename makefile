.PHONY: run
run:
	@echo "Running containers..." && \
	docker-compose down --remove-orphans && \
	docker-compose up --build

.PHONY: run-d
run-d:
	@echo "Running containers..." && \
	docker-compose down --remove-orphans && \
	docker-compose up --build -d
# .PHONY: attach
# attach:
# 	@echo "Attaching to container..." && \
# 	docker-compose up db -d && \
# 	docker-compose exec db sh
