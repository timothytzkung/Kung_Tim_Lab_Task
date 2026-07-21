import ollama


response = ollama.chat(
    model="gemma3:4b",
    messages=[{
        "role": "user",
        "content": "What's coolest dinosaur in europe?"
    }]
)

print(response["message"]["content"])