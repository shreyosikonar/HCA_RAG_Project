from google import genai
from dotenv import load_dotenv
import os
import time
import random

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Load environment variables
load_dotenv()

# Gemini Client
client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

# Models to try in order (fallback chain)
MODELS = [
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.5-pro",
]

# Embedding Model
embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5"
)

# Load ChromaDB
db = Chroma(
    persist_directory="vector_store",
    embedding_function=embedding_model
)


def call_gemini_with_retry(prompt, retries=5):
    """
    Calls Gemini API with:
    - Automatic model fallback (gemini-2.5-flash → flash-lite → pro)
    - Exponential backoff for 503 (server overloaded)
    - Linear backoff for 429 (quota exhausted)
    """
    for model_name in MODELS:
        for attempt in range(retries):
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt
                )
                return response.text

            except Exception as e:
                error_str = str(e)

                # 503 - Server overloaded (temporary) → exponential backoff
                if "503" in error_str or "UNAVAILABLE" in error_str:
                    wait = (2 ** attempt) + random.uniform(1, 3)
                    print(f"⚠️  [{model_name}] Server busy. Retrying in {wait:.1f}s... "
                          f"(Attempt {attempt + 1}/{retries})")
                    time.sleep(wait)

                # 429 - Quota exhausted → linear backoff
                elif "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                    wait = 20 * (attempt + 1)
                    print(f"⚠️  [{model_name}] Quota hit. Retrying in {wait}s... "
                          f"(Attempt {attempt + 1}/{retries})")
                    time.sleep(wait)

                # 404 - Wrong model name → skip to next model immediately
                elif "404" in error_str or "NOT_FOUND" in error_str:
                    print(f"❌  [{model_name}] Model not found. Trying next model...")
                    break  # skip retries, go to next model

                # Any other error → raise immediately
                else:
                    raise e

        print(f"❌  [{model_name}] Failed after {retries} attempts. Trying next model...\n")

    raise Exception("❌ All models failed. Please try again later or check your quota.")


# ── Main Chat Loop ──────────────────────────────────────────────────────────────
print("🤖 Hari Chand Anand & Co. Knowledge Assistant")
print("=" * 50)

while True:

    question = input("\nAsk a question (type 'exit' to quit): ").strip()

    if question.lower() == "exit":
        print("Goodbye!")
        break

    if not question:
        print("Please enter a valid question.")
        continue

    # Retrieve relevant chunks from ChromaDB
    results = db.similarity_search_with_score(question, k=3)
    results = sorted(results, key=lambda x: x[1])
    docs = [doc for doc, score in results]

    context = "\n\n".join([doc.page_content for doc in docs])

    # Build Prompt
    prompt = f"""
    You are the official knowledge assistant of Hari Chand Anand & Co.

    Answer the question using ONLY the information present in the context.
    If the answer is not in the context, say "I don't have enough information to answer that."

    Context:
    {context}

    Question:
    {question}

    Answer:
    """

    # Call Gemini with retry + fallback
    try:
        answer = call_gemini_with_retry(prompt)
        print("\nAnswer:\n")
        print(answer)

    except Exception as e:
        print(f"\n❌ Gemini API Error: {e}")
        print("💡 Tip: Your daily quota may be exhausted. Try again after 24 hours or enable billing.")