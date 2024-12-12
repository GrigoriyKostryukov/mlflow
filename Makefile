PROJECT_DIR:='src'
PYTHON_RUN=$(shell echo "python3")

install:
	$(PYTHON_RUN) -m pip install -r requirements.txt

lint/install:
	$(PYTHON_RUN) -m pip install -r ./dev-requirements.txt

lint/isort:
	@printf "%s\n" "Isort"
	$(PYTHON_RUN) -m isort --check-only --df $(PROJECT_DIR)
	@printf "\n"

lint/isort-fix:
	@printf "%s\n" "Isort"
	$(PYTHON_RUN) -m isort $(PROJECT_DIR)
	@printf "\n"

lint/black:
	@printf "%s\n" "Black"
	$(PYTHON_RUN) -m black --check  --diff $(PROJECT_DIR)
	@printf "\n"

lint/black-fix:
	@printf "%s\n" "Black"
	$(PYTHON_RUN) -m black $(PROJECT_DIR) -t py311
	@printf "\n"

lint: lint/black lint/isort

lint/fix: lint/isort-fix lint/black-fix
