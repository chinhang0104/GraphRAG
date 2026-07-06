import requests
from litellm import embedding

# Get the list of available models from the LiteLLM proxy
models = requests.get("http://localhost:4000/v1/models").json()
print(models)