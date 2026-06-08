import requests


def smoke_test():
    print("Testing Ollama connection...")

    payload = {
        "model": "llama3.1:latest",
        "messages": [
            {"role": "system", "content": "Respond with only pong"},
            {"role": "user", "content": "ping"}
        ],
        "stream": False
    }

    r = requests.post(
        "http://127.0.0.1:11434/api/chat",
        json=payload
    )

    print("Response:")
    print(r.json()["message"]["content"])


if __name__ == "__main__":
    smoke_test()
