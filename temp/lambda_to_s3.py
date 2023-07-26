import boto3
import uuid
import json


def upload_json_to_s3(bucket_name, json_object, file_name=None):
    # Initialize the S3 client
    s3 = boto3.client("s3")

    # Generate a unique file name if not provided
    if not file_name:
        file_name = f"{str(uuid.uuid4())}.json"

    # Convert the JSON object to a string
    json_string = json.dumps(json_object)

    try:
        # Upload the JSON string to the S3 bucket
        response = s3.put_object(Bucket=bucket_name, Key=file_name, Body=json_string)

        if response["ResponseMetadata"]["HTTPStatusCode"] == 200:
            print(f"JSON object uploaded successfully to '{bucket_name}/{file_name}'")
        else:
            print("Failed to upload the JSON object.")
    except Exception as e:
        print(f"Error: {e}")


# Example usage
if __name__ == "__main__":
    # Replace 'churnobyl' with your actual S3 bucket name
    bucket_name = "churnobyl"

    # Replace this with your JSON object
    json_object = {"key1": "value1", "key2": "value2", "key3": "value3"}

    # Upload the JSON object to S3
    upload_json_to_s3(bucket_name, json_object)
