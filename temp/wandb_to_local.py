from pathlib import Path
import typing as t
import wandb
import pickle as pkl

_ = wandb.login(key="50204b827471013bd142dfbf8a54c19e25144551")


def _custom_combiner(feature, category) -> str:
    """
    Creates custom column name for every category

    Args:
        feature (str): column name
        category (str): name of the category

    Returns:
        str: column name
    """
    return str(feature) + "_" + type(category).__name__ + "_" + str(category)


def load_artifacts() -> t.Dict:
    wand_api = wandb.Api()
    preprocessor_artifact = wand_api.artifact(
        "ishandandekar/Churnobyl/churnobyl-ohe-oe-stand:latest", type="preprocessors"
    )
    model_artifact = wand_api.artifact(
        "ishandandekar/model-registry/churnobyl-binary-clf:best", type="model"
    )
    preprocessor_path = preprocessor_artifact.download(root=".")
    encoder_oe_path = Path(preprocessor_path) / "encoder_oe_.pkl"
    print(encoder_oe_path.absolute())
    encoder_ohe_path = Path(preprocessor_path) / "encoder_ohe_.pkl"
    scaler_standard_path = Path(preprocessor_path) / "scaler_standard_.pkl"
    target_encoder_path = Path(preprocessor_path) / "target_encoder_.pkl"
    model_artifact_dir = model_artifact.download(root=".")
    print(model_artifact_dir)
    models: t.List[Path] = list(Path(model_artifact_dir).glob("*best_.pkl"))
    print([path.absolute for path in models])
    # assert models, "No models found"
    assert len(models) == 1, "More than one model found"
    model = models[0]
    model_path = Path(model_artifact_dir) / model
    artifacts = {
        "model": model_path,
        "encoder_oe": encoder_oe_path,
        "encoder_ohe": encoder_ohe_path,
        "scaler_standard": scaler_standard_path,
        "target_encoder": target_encoder_path,
    }
    return artifacts


def unpickle_artifacts(artifacts: t.Dict) -> t.Dict:
    """
    Unpickle artifacts
    """
    artifacts: t.Dict = {k: pkl.load(open(v, "rb")) for k, v in artifacts.items()}
    return artifacts


artifacts = load_artifacts()
artifacts = unpickle_artifacts(artifacts=artifacts)
