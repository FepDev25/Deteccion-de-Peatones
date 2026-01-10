import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    def __init__(self):
        self.telegram_token = os.getenv('TELEGRAM_TOKEN')
        self.telegram_chat_id = os.getenv('TELEGRAM_CHAT_ID')
        
        if not self.telegram_token or not self.telegram_chat_id:
            raise ValueError("Missing required environment variables")
