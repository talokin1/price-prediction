import requests

# Define the API endpoint
url = "http://127.0.0.1:5000/predict"

# Define the test input data (make sure the structure matches your FEATURES list)
test_data = {
    "event": 1,
    "section": 101,
    "row": 5,
    "seat": 12,
    "quantity": 2,
    "min_option": 1,
    "hours": 48,
    "confidence": 50,  # Optional
}

# Send a POST request with JSON data
response = requests.post(url, json=test_data)

# Print the response
if response.status_code == 200:
    print("Response:", response.json())
else:
    print("Error:", response.status_code, response.text)
