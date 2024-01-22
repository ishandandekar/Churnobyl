import requests
import json

CURL_COMMAND = """
curl -H "Content-Type: application/json" -d '{
  "concavity_mean": 0.3001,
  "concave_points_mean": 0.1471,
  "perimeter_se": 8.589,
  "area_se": 153.4,
  "texture_worst": 17.33,
  "area_worst": 2019.0
}' -XPOST http://0.0.0.0:8000/predict
"""

# Define the request payload as a Python dictionary
payload = {
    "concavity_mean": 0.3001,
    "concave_points_mean": 0.1471,
    "perimeter_se": 8.589,
    "area_se": 153.4,
    "texture_worst": 17.33,
    "area_worst": 2019.0,
}

# Convert the payload to a JSON string
payload_json = json.dumps(payload)

# Set the headers
headers = {"Content-Type": "application/json"}

# Define the API endpoint URL
url = "http://0.0.0.0:8000/predict"

# Make the POST request
response = requests.post(url, data=payload_json, headers=headers)

# Check the response status code and content
if response.status_code == 200:
    print("Request was successful. Response:")
    print(response.text)
else:
    print(f"Request failed with status code {response.status_code}.")
    print(response.text)
