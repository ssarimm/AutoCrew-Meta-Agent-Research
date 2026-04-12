"""
Displays remaining Groq API rate-limit quota by reading response headers
from a minimal (1-token) API call.

Called at the start of the pipeline and before the MRM phase so the user
can see how many tokens/requests remain before starting a long run.
"""
import os


def show_groq_limits(label: str = "", model: str = None) -> None:
    """
    Print current Groq rate-limit status to the terminal.
    No-op if GROQ_API_KEY is not set (i.e. not using Groq).

    Args:
        label: descriptive label shown in the heading
        model: the Groq model ID being used (e.g. 'groq/meta-llama/llama-4-scout-17b-16e-instruct').
               Rate limits are model-specific, so passing the correct model shows accurate limits.
               Defaults to llama-3.3-70b-versatile if not provided.
    """
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return

    # Strip the 'groq/' prefix if present (Groq REST API uses bare model names)
    groq_model = (model or "llama-3.3-70b-versatile").replace("groq/", "")

    heading = f"GROQ API LIMITS — {label}" if label else "GROQ API LIMITS"
    print("\n" + "-" * 52)
    print(f"  {heading}  [{groq_model}]")
    print("-" * 52)

    try:
        import httpx

        resp = httpx.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": groq_model,
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 1,
            },
            timeout=12,
        )

        h = resp.headers
        rem_req  = h.get("x-ratelimit-remaining-requests", "?")
        lim_req  = h.get("x-ratelimit-limit-requests",     "?")
        rem_tok  = h.get("x-ratelimit-remaining-tokens",   "?")
        lim_tok  = h.get("x-ratelimit-limit-tokens",       "?")
        rst_req  = h.get("x-ratelimit-reset-requests",     "?")
        rst_tok  = h.get("x-ratelimit-reset-tokens",       "?")

        print(f"  Requests : {rem_req:>6} remaining / {lim_req:<6} limit   resets in {rst_req}")
        print(f"  Tokens   : {rem_tok:>6} remaining / {lim_tok:<6} limit   resets in {rst_tok}")

        # Warn if tokens are getting low
        try:
            pct = int(rem_tok) / int(lim_tok) * 100
            if pct < 20:
                print(f"\n  ⚠  WARNING: only {pct:.0f}% of daily tokens remain.")
        except (ValueError, ZeroDivisionError):
            pass

    except ImportError:
        print("  httpx not installed — cannot fetch live limits.")
        print("  Check manually at: console.groq.com → Usage")
    except Exception as e:
        print(f"  Could not fetch limits: {e}")

    print("-" * 52)
