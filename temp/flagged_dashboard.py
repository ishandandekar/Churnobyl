import streamlit as st
import boto3
import plotly.express as px
import pandas as pd
from datetime import datetime, timedelta


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


# Function to get timestamps of prediction objects
def get_prediction_timestamps(bucket_name, folder_name):
    s3 = boto3.client("s3")
    response = s3.list_objects_v2(Bucket=bucket_name, Prefix=folder_name)
    timestamps = []
    if "Contents" in response:
        for obj in response["Contents"]:
            timestamps.append(
                obj["LastModified"].replace(tzinfo=None)
            )  # Convert to regular datetime object
    return timestamps


# Function to create a Plotly chart showing the frequency of predictions on a daily basis
def create_daily_frequency_chart(timestamps):
    df = pd.DataFrame({"timestamps": timestamps})
    df["timestamps"] = pd.to_datetime(df["timestamps"])
    df = df[
        df["timestamps"] >= datetime.now() - timedelta(days=365)
    ]  # Filter data for the past year
    df["date"] = df["timestamps"].dt.date
    daily_counts = df["date"].value_counts().sort_index()
    fig = px.bar(
        x=daily_counts.index,
        y=daily_counts.values,
        labels={"x": "Date", "y": "Frequency"},
        title="Predictions Frequency on a Daily Basis",
    )
    return fig


# Streamlit App
def main():
    # Set up your Streamlit dashboard
    st.set_page_config(
        page_title="Churnobyl Predictions Dashboard",
        page_icon=":bar_chart:",
    )

    # Set up your AWS credentials (ensure you have the necessary IAM permissions)
    boto3.setup_default_session(
        aws_access_key_id="churnobyl_s3_ACCESS_KEY_ID",
        aws_secret_access_key="churnobyl_s3_SECRET_ACCESS_KEY",
        region_name="churnobyl_s3_REGION",
    )

    # Title and description
    st.title("Churnobyl Predictions Dashboard")

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
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Number of Predictions")
        st.metric(label="", value=f"{num_predictions}")
        st.subheader("Prediction Objects\n")
        st.text("\n".join(prediction_objects))
        # Get timestamps of prediction objects
        prediction_timestamps = get_prediction_timestamps(
            bucket_name, folder_name_predictions
        )

        # Create a Plotly chart to visualize the frequency of predictions on a daily basis
        fig = create_daily_frequency_chart(prediction_timestamps)
        st.plotly_chart(fig)

    with col2:
        st.subheader("Number of Flagged Items")
        st.metric(label="", value=f"{num_flagged}")
        st.subheader("Flagged Objects\n")
        st.text("\n".join(flagged_objects))

    # You can add more gauge metrics or charts as needed


if __name__ == "__main__":
    main()
