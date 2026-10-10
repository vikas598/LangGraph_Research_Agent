# from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_core.rate_limiters import InMemoryRateLimiter

from dotenv import load_dotenv

load_dotenv()

# Groq free tier: 8,000 tokens/min. With ~1,500 tokens per call, ~5 calls/min stays under it.
rate_limiter = InMemoryRateLimiter(
    requests_per_second=0.08,   # ~5 per minute
    check_every_n_seconds=0.1,
    max_bucket_size=1,
)

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
    max_retries=5,          # waits and retries after a 429 instead of failing
    max_tokens=1024,        # caps the output size Groq reserves per request
    reasoning_effort="low", # gpt-oss is a reasoning model; less thinking = fewer tokens
    rate_limiter=rate_limiter,
)