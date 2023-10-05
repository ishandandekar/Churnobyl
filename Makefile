hello:
	@echo "hello world"

install_reqs:
	if [ -f requirements.txt ]; then pip install -r requirements.txt; fi

test:
	pytest --ignore=temp

setup_workspace:
	wandb login "$WANDB_API"
	echo "Wandb login successful"
	prefect cloud login -k $API_KEY --workspace ishandandekar/churnobyl