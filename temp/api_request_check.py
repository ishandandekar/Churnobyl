import requests

url = "http://localhost:8000"

try:
    response = requests.get(url)
    if response.status_code == 200:
        print("Request was successful!")
        print("Response content:")
        print(response.text)
    else:
        print(f"Request failed with status code: {response.status_code}")
except requests.exceptions.RequestException as e:
    print("An error occurred:", e)
