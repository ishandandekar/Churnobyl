"""
This script contains all the data utility functions.
"""

from pathlib import Path
from io import StringIO
import requests
import pandas as pd
import boto3


class DataDreamer:
    @staticmethod
    def load_csv_from_dir(dir: Path, columns: list) -> pd.DataFrame:
        if dir.is_dir():
            data = pd.DataFrame(columns=columns)
            paths = dir.glob("*.csv")
            if len(paths) == 0:
                raise Exception(f"No `.csv` present in the directory {dir}")
            for path in paths:
                df = pd.read_csv(path)
                if data.columns != df.columns:
                    raise Exception(f"Column mismatch error for `.csv`: {path}")
                data = pd.concat([data, df])
            return data
        else:
            raise Exception("Path provided is not a directory")

    @staticmethod
    def load_csv_from_url(url: str, columns: list) -> pd.DataFrame:
        if not url.endswith(".csv"):
            raise Exception(f"The url is not of a `.csv`: {url}")
        response = requests.get(url)
        response.raise_for_status()
        csv_file = response.content
        df = pd.read_csv(pd.compat.StringIO(csv_file.decode("utf-8")))
        if df.columns.to_list() != columns:
            raise Exception(f"Column mismatch error for `.csv`: {url}")
        return df

    @staticmethod
    def load_csv_from_aws_s3(
        bucket_name: str, region: str, session: boto3.Session
    ) -> pd.DataFrame:
        # How to create a session?
        # session = boto3.Session(
        #     aws_access_key_id='your-access-key',
        #     aws_secret_access_key='your-secret-key',
        #     region_name='your-aws-region'
        # )
        s3_client = session.client("s3")
        response = s3_client.list_objects_v2(Bucket=bucket_name)
        dataframes: list = list()
        for obj in response["Contents"]:
            key: str = obj["Key"]
            if key.endswith(".csv"):
                response = requests.get(
                    f"https://{bucket_name}.s3.{region}.amazonaws.com/{key}"
                )
                response.raise_for_status()
                csv_content = response.content.decode("utf-8")

                df = pd.read_csv(StringIO(csv_content))

                dataframes.append(df)
        combined_df = pd.concat(dataframes, ignore_index=True)
        return combined_df
