import boto3
import pandas as pd

# How to create a session?
# session = boto3.Session(
#     aws_access_key_id='your-access-key',
#     aws_secret_access_key='your-secret-key',
#     region_name='your-aws-region'
# )


def load_csvs_to_dataframe(bucket_name, folder_path, session: boto3.Session):
    try:
        # Create the S3 URL
        s3_url = f"s3://{bucket_name}/{folder_path}"

        # List all objects in the specified folder using Boto3
        s3_client = session.client("s3")
        objects = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=folder_path)

        dataframes = []

        # Load .csv files from S3 directly into dataframes
        for obj in objects.get("Contents", []):
            key = obj["Key"]
            if key.endswith(".csv"):
                file_url = f"{s3_url}/{key}"
                try:
                    dataframe = pd.read_csv(file_url)
                    dataframes.append(dataframe)
                    print(f"Loaded: {key}")
                except Exception as e:
                    print(f"Error loading {key}: {e}")

        # Concatenate all dataframes
        if dataframes:
            concatenated_df = pd.concat(dataframes, ignore_index=True)
            return concatenated_df
        else:
            return None

    except Exception as e:
        print(f"An error occurred: {e}")
        return None


if __name__ == "__main__":
    aws_access_key = input("Enter your AWS access key: ")
    aws_secret_key = input("Enter your AWS secret key: ")
    bucket_name = input("Enter the S3 bucket name: ")
    folder_path = input(
        "Enter the folder path in the bucket (e.g., 'folder/subfolder'): "
    )

    try:
        concatenated_df = load_csvs_to_dataframe(
            bucket_name, folder_path, aws_access_key, aws_secret_key
        )
        if concatenated_df is not None:
            print("Concatenated DataFrame:")
            print(concatenated_df)
        else:
            print("No CSV files were found in the specified folder")
    except:
        ...
