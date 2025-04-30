import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key=os.environ.get("OPENROUTER_API_KEY"),
)

completion = client.chat.completions.create(
  extra_body={},
  model="qwen/qwen3-235b-a22b:free",
  messages=[
    {
      "role": "user",
      "content": "人生の意味は何ですか？"
    }
  ]
)
print(completion.choices[0].message.content)