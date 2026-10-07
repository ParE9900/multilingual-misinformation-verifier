from src.config import get_config
config = get_config()

# Using official Groq SDK and latest confirmed models
GROQ_MODEL = "llama-3.3-70b-versatile"
GEMINI_MODEL = "gemini-3.5-flash"

print(f"Using Groq Model: {GROQ_MODEL}")
print("--- Direct Groq Test (Official SDK) ---")
try:
    from groq import Groq
    client = Groq(api_key=config.GROQ_API_KEY)
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": "Decompose this into 2 sub-claims. Output JSON."},
            {"role": "user", "content": "The Earth is flat."}
        ],
        temperature=0.0
    )
    print("Groq Success:", response.choices[0].message.content)
except Exception as e:
    print(f"Groq Raw Error: {type(e).__name__}: {e}")

print(f"\nUsing Gemini Model: {GEMINI_MODEL}")
print("--- Direct Gemini Test ---")
try:
    from google import genai
    client = genai.Client(api_key=config.GEMINI_API_KEY)
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents="Is the Earth flat?",
        config={"tools": [{"google_search": {}}]}
    )
    print("Gemini Success:", response.text)
except Exception as e:
    print(f"Gemini Raw Error: {type(e).__name__}: {e}")