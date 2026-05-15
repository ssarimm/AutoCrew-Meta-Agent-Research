import os

os.environ["CREWAI_TELEMETRY"] = "false"
os.environ["CREWAI_TRACE"] = "false"
os.environ["CREWAI_LOG_LEVEL"] = "ERROR"

from dotenv import load_dotenv
from crewai import LLM
from langchain_community.chat_models import FakeListChatModel

load_dotenv(override=True)

import litellm
litellm.drop_params = True
litellm.num_retries = 30
litellm.retry_wait_time = 70

if "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = "NA"


# ---------------------------
# OPENROUTER SAFE WRAPPER
# ---------------------------
def create_openrouter_llm(model, api_key):
    try:
        return LLM(
            model=model,
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            temperature=0.2,
            timeout=120,
            max_tokens=1000
        )
    except Exception as e:
        print(f"OpenRouter failed: {e}")
        print("Fallback → GPT-4o-mini")

        return LLM(
            model="openai/gpt-4o-mini",
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            temperature=0.2,
            timeout=120,
            max_tokens=1000
        )


def get_llm(choice=None):

    # --- GROQ ---
    if choice == "1":
        print("Using GROQ Llama 3.3 70B...")
        api_key = os.getenv("GROQ_API_KEY") or input("Groq API Key: ")
        os.environ["GROQ_API_KEY"] = api_key

        return LLM(
            model="groq/llama-3.3-70b-versatile",
            api_key=api_key,
            temperature=0.2,
            timeout=120
        )

    # --- GEMINI (DIRECT - FIXED) ---
    elif choice == "2":
        print("Using Google Gemini...")

        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            api_key = input("Enter Gemini API Key: ").strip()
            os.environ["GOOGLE_API_KEY"] = api_key

        return LLM(
            model="gemini/gemini-3-flash-preview",
            api_key=api_key,
            temperature=0.2,
            timeout=120
        )

    # --- OLLAMA ---
    elif choice == "3":
        ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        ollama_model_name = os.getenv("OLLAMA_MODEL_NAME", "llama3.1:8b")

        print(f"Using OLLAMA {ollama_model_name}")

        return LLM(
            model=f"ollama/{ollama_model_name}",
            base_url=f"{ollama_base_url}/v1",
            api_key="ollama",
            temperature=0.2,
            timeout=300
        )

    # --- MOCK ---
    elif choice == "4":
        print("Using MOCK LLM")

        fake_llm = FakeListChatModel(responses=[
            "Data loaded",
            "Training complete",
            "Approved"
        ])

        class MockLLM:
            def __init__(self, fake_llm):
                self.fake_llm = fake_llm
                self.model = "mock"

            def invoke(self, messages):
                prompt = "\n".join([m.content for m in messages if hasattr(m, "content")])
                return self.fake_llm.invoke(prompt)

        return MockLLM(fake_llm)

    # --- GROQ SCOUT ---
    elif choice == "5":
        print("Using GROQ Llama 4 Scout...")

        api_key = os.getenv("GROQ_API_KEY") or input("Groq API Key: ")
        os.environ["GROQ_API_KEY"] = api_key

        return LLM(
            model="groq/meta-llama/llama-4-scout-17b-16e-instruct",
            api_key=api_key,
            temperature=0.2,
            timeout=120
        )

    # --- GROQ FAST ---
    elif choice == "6":
        print("Using GROQ 3.1 8B Instant...")

        api_key = os.getenv("GROQ_API_KEY") or input("Groq API Key: ")
        os.environ["GROQ_API_KEY"] = api_key

        return LLM(
            model="groq/llama-3.1-8b-instant",
            api_key=api_key,
            temperature=0.2,
            timeout=120
        )

    # --- OPENROUTER ---
    elif choice == "7":
        print("\nSelect OpenRouter Model:")
        print("1. Qwen 2.5 72B (Reasoning)")
        print("2. GPT-4o-mini (Stable ⭐)")
        print("3. Gemini 2.0 Flash (Fast ⚡)")
    
        sub_choice = input("Enter choice (1-3): ").strip()
    
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            api_key = input("Enter OpenRouter API Key: ").strip()
            os.environ["OPENROUTER_API_KEY"] = api_key
    
        if sub_choice == "1":
            model = "qwen/qwen-2.5-72b-instruct"
    
        elif sub_choice == "2":
            model = "openai/gpt-4o-mini"
    
        elif sub_choice == "3":
            model = "google/gemini-2.0-flash"
    
        else:
            print("Invalid choice → default GPT-4o-mini")
            model = "openai/gpt-4o-mini"
    
        return create_openrouter_llm(model, api_key)


def select_llm_interactive():
    print("\nSelect Provider:")
    print("1. Groq 70B")
    print("2. Gemini ⭐")
    print("3. Ollama")
    print("4. Mock")
    print("5. Groq Scout")
    print("6. Groq 8B Fast")
    print("7. OpenRouter")

    choice = input("Choice (1-7): ").strip()

    llm = get_llm(choice)
    return llm, choice