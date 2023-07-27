import streamlit as st
import boto3


# Function to count the number of objects in a folder
def count_objects_in_folder(bucket_name, folder_name):
    s3 = boto3.client("s3")
    response = s3.list_objects_v2(Bucket=bucket_name, Prefix=folder_name)
    if "Contents" in response:
        return len(response["Contents"])
    return 0


# Function to list the objects in a folder
def list_objects_in_folder(bucket_name, folder_name):
    s3 = boto3.client("s3")
    response = s3.list_objects_v2(Bucket=bucket_name, Prefix=folder_name)
    if "Contents" in response:
        return [obj["Key"] for obj in response["Contents"]]
    return []


# Streamlit App
def main():
    # Set up your Streamlit dashboard
    st.set_page_config(
        page_title="Churnobyl Predictions Dashboard",
        page_icon=":bar_chart:",
        layout="wide",
    )

    # Set up your AWS credentials (ensure you have the necessary IAM permissions)
    boto3.setup_default_session(
        aws_access_key_id="churnobyl_s3_ACCESS_KEY_ID",
        aws_secret_access_key="churnobyl_s3_SECRET_ACCESS_KEY",
        region_name="churnobyl_s3_REGION",
    )

    # Title and description
    st.title("Churnobyl Predictions Dashboard")
    st.write(
        "Metrics for predictions made in the 'predict_logs' folder and 'flagged' folder."
    )

    # Replace 'YOUR_BUCKET_NAME' with your actual bucket name
    bucket_name = "churnobyl"
    folder_name_predictions = "api_logs/predict_logs"
    folder_name_flagged = "flagged"

    # Get the number of predictions and list prediction objects
    num_predictions = count_objects_in_folder(bucket_name, folder_name_predictions) - 1
    prediction_objects = list_objects_in_folder(bucket_name, folder_name_predictions)
    prediction_objects.remove(f"{folder_name_predictions}/")

    # Get the number of flagged items and list flagged objects
    num_flagged = count_objects_in_folder(bucket_name, folder_name_flagged) - 1
    flagged_objects = list_objects_in_folder(bucket_name, folder_name_flagged)
    flagged_objects.remove(f"{folder_name_flagged}/")

    # Display the gauge metrics and list of objects
    st.subheader("Number of Predictions")
    st.text(f"Total predictions: {num_predictions}")
    st.subheader("Prediction Objects")
    st.text("\n".join(prediction_objects))

    st.subheader("Number of Flagged Items")
    st.text(f"Total flagged items: {num_flagged}")
    st.subheader("Flagged Objects")
    st.text("\n".join(flagged_objects))

    # You can add more gauge metrics or charts as needed


if __name__ == "__main__":
    main()
