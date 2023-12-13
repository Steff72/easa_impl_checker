import openai

response = openai.embeddings.create(
    input="Edelweiss is great!", model="text-embedding-ada-002"
)

print(response)
