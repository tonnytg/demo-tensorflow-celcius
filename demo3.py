import requests
import json

# Endpoint da API
url = "http://localhost:11434/api/generate"

# Dados da solicitação
payload = {
    "model": "llama3",
    "prompt": "Why is the sky blue?"
}

# Cabeçalhos da solicitação
headers = {
    "Content-Type": "application/json"
}

# Fazendo a solicitação POST
response = requests.post(url, data=json.dumps(payload), headers=headers)

# Verificando o status da resposta
if response.status_code == 200:
    # Imprimindo a resposta
    result = response.json()
    print("Resposta da API:")
    print(json.dumps(result, indent=4))
else:
    print(f"Erro: {response.status_code}")
    print(response.text)
