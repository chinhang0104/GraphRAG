import os
from dotenv import load_dotenv
import litellm

load_dotenv()

embed = litellm.embedding(
    model="llamafile/mxbai-embed",
    input=["GraphRAG embedding test"],
)

print(embed)

chat = litellm.completion(
    model="groq/llama-3.3-70b-versatile",
    messages=[{"role": "user", "content": "Say hello in one sentence."}],
    api_key=os.environ["GROQ_API_KEY"],
)

print(chat.choices[0].message.content)