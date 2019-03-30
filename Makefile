.DEFAULT_GOAL := all

.PHONY: install
install:
	pipenv install --dev

.PHONY: lint
lint:
	flake8 .

.PHONY: test
test:
	pytest -W ignore

.PHONY: all
all: lint test
