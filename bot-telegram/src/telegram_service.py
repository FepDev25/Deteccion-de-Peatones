import requests

class TelegramService:
    def __init__(self, token, chat_id):
        self.token = token
        self.chat_id = chat_id
        self.api_url = f"https://api.telegram.org/bot{token}/sendPhoto"
    
    def send_notification(self, image_bytes, status_text):
        try:
            caption = f"Estado reporte: {status_text}"
            payload = {'chat_id': self.chat_id, 'caption': caption}
            files = {'photo': ('analysis.jpg', image_bytes, 'image/jpeg')}
            
            response = requests.post(self.api_url, data=payload, files=files)
            
            if response.status_code == 200:
                print("[TELEGRAM] Notificacion enviada")
            else:
                print(f"[TELEGRAM] Error: {response.status_code}")
                
        except Exception as e:
            print(f"[TELEGRAM] Error: {e}")
