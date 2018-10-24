import json

with open('result_cluster.json') as f:
    data = json.load(f)

print(data)