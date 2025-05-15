import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL")
SECRET_KEY = os.getenv("SECRET_KEY")
AUTHENTICATION_REQUIRED = os.getenv("AUTHENTICATION_REQUIRED", "False").lower() == "true"