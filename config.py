import os
from dotenv import load_dotenv

load_dotenv()

FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:3000")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
