import requests
import sys

if len(sys.argv) == 1:
    url = 'http://0.0.0.0:9696/predict'
else:
    url = f'{sys.argv[1]}/predict'

query_test = [
    {
     "query" : "Is it Possible to use AWS insted of GCP?"
    }
]

print("=========================== TEST QUERY ANSWERS ===========================")
answer = requests.post(url, json=query_test[0]).json()
print(f"Response {answer}")