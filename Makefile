hello:
	@echo "hello world"

install_reqs:
	if [ -f requirements.txt ]; then pip install -r requirements.txt; fi

test:
	pytest --ignore=temp