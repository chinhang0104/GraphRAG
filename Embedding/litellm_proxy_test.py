import openai

# Start the LiteLLM proxy server in a separate terminal before running this test:
# litellm --config config.yaml

# Connect to the LiteLLM Proxy instead of OpenAI
client = openai.OpenAI(
    api_key="sk-1234567890", # Dummy key, LiteLLM accepts anything here by default
    base_url="http://localhost:4000" 
)

response = client.embeddings.create(
    model="mxbai-embed", # The alias you defined in config.yaml
    input=["Testing the proxy embedding."]
)

print(response)