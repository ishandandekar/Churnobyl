import wandb
from pathlib import Path

api = wandb.Api()

artifact = api.artifact(
    "ishandandekar/Churnobyl/churnobyl-ohe-oe-stand:latest", type="preprocessors"
)
# artifact.download(root=".")
preprocessor_path = artifact.download()
print(type(preprocessor_path))
model_artifact = api.artifact(
    "ishandandekar/model-registry/churnobyl-binary-clf:latest", type="model"
)
model_artifact_dir = model_artifact.download()
print(model_artifact_dir)
models = list(Path(model_artifact_dir).glob("*.pkl"))
assert models, "No models found"
assert len(models) == 1, "More than one model found"
model = models[0]
model_path = Path(model_artifact_dir) / model
print(model_path)
