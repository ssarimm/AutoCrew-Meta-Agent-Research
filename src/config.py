import os
from dotenv import load_dotenv
from crewai import LLM
from langchain_community.chat_models import FakeListChatModel
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

import litellm
litellm.drop_params = True
litellm.num_retries = 30
litellm.retry_wait_time = 45

# Prevent crewai from erroring when OpenAI key is missing
if "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = "NA"


def get_llm(choice=None):
    """
    Returns an LLM instance based on user choice.
    Used by BOTH manual and auto paths.
    """

    # --- GROQ ---
    if choice == "1":
        print("Using GROQ (Llama 3.3 70B)...")
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            api_key = input("Enter Groq API Key (gsk_...): ")
            os.environ["GROQ_API_KEY"] = api_key

        return LLM(
            model="groq/llama-3.3-70b-versatile",
            api_key=api_key,
            temperature=0.2,
            timeout=120
        )

    # --- GOOGLE GEMINI ---
    elif choice == "2":
        print("Using GOOGLE GEMINI 2.0 Flash...")
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            api_key = input("Enter Gemini API Key: ")
            os.environ["GOOGLE_API_KEY"] = api_key

        return LLM(
            model="gemini/gemini-2.0-flash",
            api_key=api_key,
            temperature=0.2,
            timeout=120
        )

    # --- OLLAMA (LOCAL) ---
    elif choice == "3":
        ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        ollama_model_name = os.getenv("OLLAMA_MODEL_NAME", "llama3.1:8b")

        print(f"Using OLLAMA ({ollama_model_name} @ {ollama_base_url})...")

        return LLM(
            model=f"ollama/{ollama_model_name}",
            base_url=f"{ollama_base_url}/v1",
            api_key="ollama",
            temperature=0.2,
            timeout=300
        )

    # --- MOCK LLM (TESTING) ---
    elif choice == "4":
        print("Using MOCK LLM (Testing Mode)...")
        responses = [
            "Data Extraction: Loaded data successfully.",
            "Model Training: Accuracy 0.95.",
            "MRM: Approved."
        ]
        return FakeListChatModel(responses=responses)

    else:
        print("ERROR: Invalid choice. Using MOCK LLM as fallback.")
        responses = ["Fallback: Mock Output"]
        return FakeListChatModel(responses=responses)


def select_llm_interactive():
    """Interactive LLM selection menu. Returns (llm, choice_str)."""
    print("\nSelect LLM Provider:")
    print("1. Groq (Llama 3.3 70B)")
    print("2. Google Gemini 2.0 Flash")
    print("3. Ollama (Local LLM)")
    print("4. Mock LLM (Testing)")
    choice = input("Choice (1-4): ").strip()

    if choice not in ["1", "2", "3", "4"]:
        print("Invalid choice. Exiting.")
        return None, None

    llm = get_llm(choice)
    return llm, choice
