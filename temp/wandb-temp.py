import wandb

api = wandb.Api()

artifact = api.artifact(
    "ishandandekar/Churnobyl/churnobyl-ohe-oe-stand:latest", type="preprocessors"
)

artifact.download(root=".")
