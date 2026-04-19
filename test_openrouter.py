import urllib.request, json, os
from dotenv import load_dotenv
load_dotenv('.env')

url = 'https://openrouter.ai/api/v1/embeddings'
headers = {
    'Authorization': f'Bearer {os.getenv("OPENROUTER_API_KEY")}',
    'Content-Type': 'application/json',
    'User-Agent': 'UFM-Admission-Bot/1.0',
}
data = {
    'model': 'baai/bge-m3',
    'input': ['This is a test'],
    'dimensions': 1024,
}

req = urllib.request.Request(url, data=json.dumps(data).encode('utf-8'), headers=headers, method='POST')
try:
    with urllib.request.urlopen(req) as resp:
        print('HTTP', resp.status)
        print(resp.read().decode('utf-8'))
except urllib.error.HTTPError as e:
    print('HTTP ERROR', e.code)
    print(e.read().decode('utf-8'))
