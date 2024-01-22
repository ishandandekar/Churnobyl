"""
This script contains all the data utility functions.
"""
import typing as t
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import boto3
import cloudpickle as cpickle
import numpy as np
import pandera as pa
import polars as pl
from box import Box
from scipy.sparse import spmatrix
from sklearn import compose, preprocessing
from sklearn.model_selection import train_test_split
from src.exceptions import ConfigValidationError

# Dictionary containing conditions for columns for validation
checks: t.Dict[str, t.List[pa.Check]] = {
    "customerID": [],
    "gender": [pa.Check.isin(["Male", "Female"])],
    "SeniorCitizen": [pa.Check.isin([0, 1])],
    "Partner": [pa.Check.isin(["Yes", "No"])],
    "Dependents": [pa.Check.isin(["Yes", "No"])],
    "tenure": [
        pa.Check.greater_than_or_equal_to(min_value=0.0),
    ],
    "PhoneService": [pa.Check.isin(["No", "Yes"])],
    "MultipleLines": [pa.Check.isin(["No", "Yes", "No phone service"])],
    "InternetService": [pa.Check.isin(["DSL", "Fiber optic", "No"])],
    "OnlineSecurity": [pa.Check.isin(["No", "Yes", "No internet service"])],
    "OnlineBackup": [pa.Check.isin(["Yes", "No", "No internet service"])],
    "DeviceProtection": [pa.Check.isin(["No", "Yes", "No internet service"])],
    "TechSupport": [pa.Check.isin(["No", "Yes", "No internet service"])],
    "StreamingTV": [pa.Check.isin(["No", "Yes", "No internet service"])],
    "StreamingMovies": [pa.Check.isin(["No", "Yes", "No internet service"])],
    "Contract": [pa.Check.isin(["Month-to-month", "One year", "Two year"])],
    "PaperlessBilling": [pa.Check.isin(["Yes", "No"])],
    "PaymentMethod": [
        pa.Check.isin(
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ]
        )
    ],
    "MonthlyCharges": [],
    "TotalCharges": [],
    "Churn": [pa.Check.isin(["No", "Yes"])],
}

# Data schema for validating and checking the new data
DataSchema = pa.DataFrameSchema(
    columns={
        "customerID": pa.Column(
            dtype="object",
            checks=checks.get("customerID"),
            nullable=False,
            unique=False,
            coerce=False,
            required=True,
            regex=False,
            description=None,
            title=None,
        ),
        "gender": pa.Column(
            dtype="object",
            checks=checks.get("gender"),
            nullable=False,
            unique=False,
            coerce=False,
            required=True,
            regex=False,
            description=None,
            title=None,
        ),
        "SeniorCitizen": pa.Column(
            dtype="int64",
            checks=checks.get("SeniorCitizen"),
            nullable=False,
            unique=False,
            coerce=False,
            required=True,
            regex=False,
            description=None,
            title=None,
        ),
        "Partner": pa.Column(
            dtype="object",
            checks=checks.get("Partner"),
            nullable=False,
            unique=False,
            coerce=False,
            required=True,
            regex=False,
            description=None,
            title=None,
        ),
        "Dependents": pa.Column(
            dtype="object",
            checks=checks.get("Dependents"),
            nullable=False,
            unique=False,
            coerce=False,
            required=True,
            regex=False,
            description=None,
            title=None,
        ),
        "tenure": pa.Column(
            dtype="int64",
            checks=checks.get("tenure"),
            nullable=False,
            unique=False,
            coerce=False,
            required=True,
            regex=False,
            description=None,
            title=None,
        ),
        "PhoneService": pa.Column(
            dtype="object",
            checks=checks.get("PhoneService"),
            nullable=False,
            unique=False,
            coerce=False,
            required=True,
            regex=False,
            description=None,
            title=None,
        ),
        "MultipleLines": pa.Column(
            dtype="object",
            checks=checks.get("MultipleLines"),
            nullable=False,
            unique=False,
            coerce=False,
            required=True,
            regex=False,
            description=None,
            title=None,
        ),
        "InternetService": pa.Column(
            dtype="object",
            checks=checks.get("InternetService"),
            nullable=False,
            unique=False,
            coerce=False,
            required=True,
            regex=False,
            description=None,
            title=None,
        ),
        "OnlineSecurity": pa.Column(
            dtype="object",
            checks=checks.get("OnlineSecurity"),
            nullable=False,
            unique=False,
            coerce=False,
            required=True,
            regex=False,
            description=None,
            title=None,
        ),
        "OnlineBackup": pa.Column(
            dtype="object",
            checks=checks.get("OnlineBackup"),
            nullable=False,
            unique=False,
            coerce=False,
            required=True,
            regex=False,
            description=None,
            title=None,
        ),
        "DeviceProtection": pa.Column(
            dtype="object",
            checks=checks.get("DeviceProtection"),
            nullable=False,
            unique=False,
            coerce=False,
            required=True,
            regex=False,
            description=None,
            title=None,
        ),
        "TechSupport": pa.Column(
            dtype="object",
            checks=checks.get("TechSupport"),
            nullable=False,
            unique=False,
            coerce=False,
            required=True,
            regex=False,
            description=None,
            title=None,
        ),
        "StreamingTV": pa.Column(
            dtype="object",
            checks=checks.get("StreamingTV"),
            nullable=False,
            unique=False,
            coerce=False,
            required=True,
            regex=False,
            description=None,
            title=None,
        ),
        "StreamingMovies": pa.Column(
            dtype="object",
            checks=checks.get("StreamingMovies"),
            nullable=False,
            unique=False,
            coerce=False,
            required=True,
            regex=False,
            description=None,
            title=None,
        ),
        "Contract": pa.Column(
            dtype="object",
            checks=checks.get("Contract"),
            nullable=False,
            unique=False,
            coerce=False,
            required=True,
            regex=False,
            description=None,
            title=None,
        ),
        "PaperlessBilling": pa.Column(
            dtype="object",
            checks=None,
            nullable=False,
            unique=False,
            coerce=False,
            required=True,
            regex=False,
            description=None,
            title=None,
        ),
        "PaymentMethod": pa.Column(
            dtype="object",
            checks=checks.get("PaymentMethod"),
            nullable=False,
            unique=False,
            coerce=False,
            required=True,
            regex=False,
            description=None,
            title=None,
        ),
        "MonthlyCharges": pa.Column(
            dtype="float64",
            checks=checks.get("MonthlyCharges"),
            nullable=False,
            unique=False,
            coerce=False,
            required=True,
            regex=False,
            description=None,
            title=None,
        ),
        "TotalCharges": pa.Column(
            dtype="object",
            checks=checks.get("TotalCharges"),
            nullable=False,
            unique=False,
            coerce=False,
            required=True,
            regex=False,
            description=None,
            title=None,
        ),
        "Churn": pa.Column(
            dtype="object",
            checks=checks.get("Churn"),
            nullable=False,
            unique=False,
            coerce=False,
            required=True,
            regex=False,
            description=None,
            title=None,
        ),
    },
    checks=None,
    index=pa.Index(
        dtype="int64",
        checks=[],
        nullable=False,
        coerce=False,
        name=None,
        description=None,
        title=None,
    ),
    dtype=None,
    coerce=True,
    strict=False,
    name=None,
    ordered=False,
    unique=None,
    report_duplicates="all",
    unique_column_names=False,
    title=None,
    description=None,
)


class BaseDataLoaderStrategy(ABC):
    """
    Base class for data loading strategies
    """

    @abstractmethod
    def __call__(self) -> pl.DataFrame:
        """
        Load data
        """


@dataclass
class DirDataLoaderStrategy(BaseDataLoaderStrategy):
    """
    Strategy for loading `.csv` files from a directory

    Args:
        dir (t.Union[Path, str]): Directory where data files can be found
    """

    dir: t.Union[Path, str]

    def __call__(self) -> pl.DataFrame:
        """
        Function to load data

        Raises:
            FileNotFoundError: When there are no `.csv` files found
            NotADirectoryError: When arg `dir` is not a path to directory

        Returns:
            pl.DataFrame: Data from the folder as a DataFrame
        """
        if isinstance(self.dir, str):
            self.dir = Path(self.dir)
        if self.dir.is_dir():
            paths = list(self.dir.glob("*.csv"))
            if len(paths) == 0:
                raise FileNotFoundError(f"No `.csv` present in the directory {dir}")

            def _read(path) -> pl.DataFrame:
                return pl.scan_csv(
                    path, dtypes={"TotalCharges": pl.String}
                ).with_columns(
                    pl.col("TotalCharges")
                    .str.replace(pattern=" ", value="0")
                    .alias("TotalCharges")
                )

            return pl.concat(list(map(_read, paths)), how="vertical").collect()
        else:
            raise NotADirectoryError(f"{self.dir.absolute()} is not a directory")


@dataclass
class AwsS3DataLoaderStrategy(BaseDataLoaderStrategy):
    """
    Strategy to load `.csv` files from AWS S3 bucket

    Args:
        bucket_name (str): Name of the bucket
        folder_prefix (str): Name of the folder to pick data from
    """

    bucket_name: str
    folder_prefix: str

    def __call__(self) -> pl.DataFrame:
        """
        Function to load data

        Returns:
            pl.DataFrame: Data from the bucket as DataFrame
        """
        client = boto3.Session().client("s3")
        if self.folder_path == "":
            s3_url = f"s3://{self.bucket_name}"
            objects = client.list_objects_v2(Bucket=self.bucket_name)
        else:
            s3_url = f"s3://{self.bucket_name}/{self.folder_path}"
            objects = client.list_objects_v2(
                Bucket=self.bucket_name, Prefix=self.folder_path
            )

        dataframes: list[pl.LazyFrame] = list()
        for obj in objects.get("Contents", []):
            key = obj["Key"]
            if key.endswith(".csv"):
                file_url = f"{s3_url}/{key}"
                try:
                    dataframe = pl.scan_csv(
                        file_url, dtypes={"TotalCharges": pl.String}
                    ).with_columns(
                        pl.col("TotalCharges")
                        .str.replace(pattern=" ", value="0")
                        .alias("TotalCharges")
                    )
                    dataframes.append(dataframe)
                except Exception as e:
                    print(f"Error loading {key}: {e}")

        if len(dataframes) == 0:
            raise Exception(
                "No data found. Check the contents in the bucket and folder"
            )
        return pl.concat(dataframes, ignore_index=True).collect()


# Factory for all the loading strategies
DataLoaderStrategyFactory: t.Dict[str, t.Type[BaseDataLoaderStrategy]] = {
    "dir": DirDataLoaderStrategy,
    "aws_s3": AwsS3DataLoaderStrategy,
}


# Output class for better annotations
@dataclass(frozen=True)
class TransformerOutput:
    X_train: t.Annotated[t.Union[np.ndarray, spmatrix], "Training features"]
    X_test: t.Annotated[t.Union[np.ndarray, spmatrix], "Test features"]
    y_train: t.Annotated[t.Union[np.ndarray, spmatrix], "Training labels"]
    y_test: t.Annotated[t.Union[np.ndarray, spmatrix], "Test label"]


class DataEngine:
    @staticmethod
    def load(config: Box) -> pl.DataFrame:
        """
        Load data using strategy

        Args:
            config (Box): Configurations arguments for strategies

        Raises:
            Exception: Raised when there is no strategy present as specified in configuration

        Returns:
            pl.DataFrame: Data as a polars.DataFrame
        """
        if config.strategy is not None:
            return DataLoaderStrategyFactory.get(config.strategy)(**config.args)()
        else:
            raise Exception("Arguments for loading the data must be given")

    @staticmethod
    def validate(data: pl.DataFrame) -> pl.DataFrame:
        """
        Checks if data fits the appropriate conditions using a schema

        Args:
            data (pl.DataFrame): Data required to be checked

        Raises:
            Exception: When data doesn't fit the required schema

        Returns:
            pl.DataFrame: Same data if passed
        """
        try:
            DataSchema.validate(data.to_pandas(), lazy=True)
        except pa.errors.SchemaErrors as err:
            print("Schema errors and failure cases:")
            print(err.failure_cases)
            print("\nDataFrame object that failed validation:")
            print(err.data)
            raise Exception("Schema errors and failure cases")

        return data

    @staticmethod
    def split(
        config: Box,
        data: pl.DataFrame,
        seed: int,
    ) -> t.Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        """
        Split the data into training and test sets of features and labels pairs

        Args:
            config (Box): Configuration mapping
            data (pl.DataFrame): Whole data needed to be split
            seed (int): Random seed

        Returns:
            t.Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]: Tuple of train features, test features, train labels and test labels
        """
        X, y = data.drop("Churn"), data.select("Churn")
        if config.stratify:
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=config.ratio,
                random_state=seed,
                stratify=y,
            )
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=config.ratio,
                random_state=seed,
            )

        return X_train, X_test, y_train, y_test

    @staticmethod
    def transform(
        config: Box,
        X_train: pl.DataFrame,
        X_test: pl.DataFrame,
        y_train: pl.DataFrame,
        y_test: pl.DataFrame,
        artifact_dir: Path,
    ) -> TransformerOutput:
        """
        Transforms training and test data using scaling and encoding methods

        Args:
            config (Box): Configuration mapping
            X_train (pl.DataFrame): Training features
            X_test (pl.DataFrame): Test features
            y_train (pl.DataFrame): Training labels
            y_test (pl.DataFrame): Test labels
            artifact_dir (Path): Directory to store trained preprocessors

        Returns:
            TransformerOutput: Contains training and test features and labels seperately
        """
        feature_transformer = compose.ColumnTransformer(
            transformers=[
                ("num_scaler", preprocessing.StandardScaler(), config.scale),
                ("cat_ohe", preprocessing.OneHotEncoder(), config.dummy),
                ("cat_oe", preprocessing.OrdinalEncoder(), config.ordinal),
            ]
        )
        label_transformer = preprocessing.LabelBinarizer()
        feature_transformer.fit(X_train.to_pandas())
        label_transformer.fit(y_train.to_pandas())
        preprocessor_paths = ["feature_transformer.pkl", "label_binarizer.pkl"]
        preprocessors = [feature_transformer, label_transformer]

        def _save_to_pickle(
            path_: str,
            preprocessor: t.Union[
                compose.ColumnTransformer, preprocessing.LabelBinarizer
            ],
        ):
            path: Path = artifact_dir / path_
            with open(str(path), "wb") as f_out:
                cpickle.dump(preprocessor, f_out)

        # Saving trained preprocessors
        for path, preprocessor in zip(preprocessor_paths, preprocessors):
            _save_to_pickle(path, preprocessor)

        transformed_dataset = TransformerOutput(
            feature_transformer.transform(X_train.to_pandas()),
            feature_transformer.transform(X_test.to_pandas()),
            label_transformer.transform(y_train.to_pandas()),
            label_transformer.transform(y_test.to_pandas()),
        )

        # Saving transformed data into directory as an artifact
        np.savez(
            artifact_dir / "train_dataset",
            features=transformed_dataset.X_train,
            labels=transformed_dataset.y_train,
        )
        np.savez(
            artifact_dir / "test_dataset",
            features=transformed_dataset.X_test,
            labels=transformed_dataset.y_test,
        )

        return transformed_dataset


def validate_data_config(config: Box) -> None:
    """
    Validates configuration mapping specific to data

    Args:
        config (Box): Configuration mapping
    """
    data_config = config.data

    # Validating base data args
    data_stages = ["load", "split", "transform"]
    for stage in data_config.to_dict().keys():
        if stage not in data_stages:
            raise ConfigValidationError(
                f"Stage '{stage}' not in DataStages{data_stages}"
            )
    del data_stages

    # Validating load args
    load_config = data_config.load
    if load_config.strategy not in DataLoaderStrategyFactory.keys():
        raise ConfigValidationError(
            f"Strategy '{load_config.strategy}' not valid data loading strategy. Choose one of {list(DataLoaderStrategyFactory.keys())}."
        )
    del load_config

    # Validating split args
    split_config = data_config.split
    split_args = ["ratio", "stratify"]
    for arg in split_config.to_dict().keys():
        if arg not in split_args:
            raise ConfigValidationError(
                f"Unknown split argument '{arg}'. Should be {split_args}"
            )
    if float(split_config.ratio) > 1.0:
        raise ConfigValidationError(
            "Split should be a ratio with float dtype. It should be between 0.0 and 1.0"
        )
    if not isinstance(split_config.stratify, bool):
        raise ConfigValidationError("Stratify should either be true or false")
    del split_config
    del split_args

    # Validating transform args
    transform_config = data_config.transform.to_dict()
    transform_params = ["scale", "dummy", "ordinal"]

    for param in transform_config.keys():
        if param not in transform_params:
            raise ConfigValidationError(
                f"{param} not present in transformation parameters {transform_params}"
            )
    cols = list()
    for param in transform_params:
        for col in transform_config.get(param):
            cols.append(col)
    if not (len(cols) == len(set(cols))):
        raise ConfigValidationError(
            "Columns should be unique within the transform parameters"
        )
    del transform_config
    del transform_params
    del param
    del cols
