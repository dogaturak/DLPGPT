import requests


def main():
    print("Testing Ollama...")

    url = "http://127.0.0.1:11434/api/generate"

    payload = {
        "model": "llama3.1:latest",
        "prompt": "Say pong and nothing else",
        "stream": False
    }

    r = requests.post(url, json=payload, timeout=300)

    print("Status:", r.status_code)
    print("Response:", r.json()["response"])


if __name__ == "__main__":
    main()
