import os
import sys
from typing import List, Dict, Any

from openai import AzureOpenAI

# Entra ID (Azure AD) only needed if you choose token-based auth
from azure.identity import DefaultAzureCredential, get_bearer_token_provider


API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")

def build_client() -> AzureOpenAI:
    """
    Builds an AzureOpenAI client using either:
      - API key auth (AZURE_OPENAI_API_KEY), or
      - Entra ID auth via DefaultAzureCredential
    """
    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
    if not endpoint:
        sys.exit("AZURE_OPENAI_ENDPOINT is required (e.g., https://<resource>.openai.azure.com/).")

    api_key = os.environ.get("AZURE_OPENAI_API_KEY")

    if api_key:
        # API key auth (simplest)
        return AzureOpenAI(
            api_key=api_key,
            azure_endpoint=endpoint,
            api_version=API_VERSION,
        )
    else:
        # Entra ID auth using DefaultAzureCredential
        # Requires AZURE_TENANT_ID / AZURE_CLIENT_ID / AZURE_CLIENT_SECRET (or an active az login)
        token_provider = get_bearer_token_provider(
            DefaultAzureCredential(),
            "https://cognitiveservices.azure.com/.default",
        )
        return AzureOpenAI(
            azure_endpoint=endpoint,
            azure_ad_token_provider=token_provider,
            api_version=API_VERSION,
        )


def chat_once(
    client: AzureOpenAI,
    deployment: str,
    messages: List[Dict[str, Any]],
    max_tokens: int = 500,
    temperature: float = 0.7,
) -> str:
    """
    Sends one turn to the Chat Completions API and returns assistant text.
    """
    resp = client.chat.completions.create(
        model=deployment,         # IMPORTANT: this must be your *deployment name*
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=0.95,
    )
    return resp.choices[0].message.content or ""


def main():
    deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT")
    if not deployment:
        sys.exit("AZURE_OPENAI_DEPLOYMENT (your deployment name in Azure) is required.")

    client = build_client()

    # System behavior — tweak as you like
    messages: List[Dict[str, Any]] = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": "You are a helpful assistant for a CPT code project. Be concise and precise."}
            ],
        }
    ]

    print("Azure AI Foundry Chatbot (Ctrl+C to exit)\n")
    while True:
        try:
            user_text = input("You: ").strip()
            if not user_text:
                continue

            messages.append({"role": "user", "content": [{"type": "text", "text": user_text}]})

            assistant_text = chat_once(client, deployment, messages)
            messages.append({"role": "assistant", "content": [{"type": "text", "text": assistant_text}]})

            print(f"\nAssistant: {assistant_text}\n")

        except (KeyboardInterrupt, EOFError):
            print("\nBye!")
            break
        except Exception as e:
            # Print errors cleanly without a huge stack trace
            print(f"\n[Error] {e}\n", file=sys.stderr)


if __name__ == "__main__":
    main()
