import requests

response = requests.get("http://10.16.148.9:5000/predict",
                        params={'string': "omg! my hair is a disaster"})

print(response.json())

response = requests.get("http://10.16.148.9:5000/predict",
                        params={'string': "This forest is on fire"})

response = requests.get("http://10.16.148.9:5000/predict",
                        params={'string': "This floor is on fire"})

response = requests.get(
    "http://10.16.148.9:5000/predict",
    params={
        'string':
        "Me, a recent grad with no NLP experience. BERT. Learning fundamental techniques that don't require expensive hardware"
    })

print(response.json())
