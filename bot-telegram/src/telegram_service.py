import requests

class TelegramService:
    def __init__(self, token, chat_id):
        self.token = token
        self.chat_id = chat_id
        self.api_url = f"https://api.telegram.org/bot{token}/sendPhoto"
    
    def send_notification(self, image_bytes, status_text):
        try:
            caption = f" Estado: {status_text}"
            payload = {'chat_id': self.chat_id, 'caption': caption}
            
            # Asegurar que image_bytes es el tipo correcto
            if isinstance(image_bytes, bytes):
                files = {'photo': ('detection.jpg', image_bytes, 'image/jpeg')}
            else:
                # Si viene como memoryview o ndarray, convertir
                files = {'photo': ('detection.jpg', bytes(image_bytes), 'image/jpeg')}
            
            response = requests.post(self.api_url, data=payload, files=files, timeout=10)
            
            if response.status_code == 200:
                print("[TELEGRAM] Notificación enviada correctamente")
            else:
                print(f"[TELEGRAM] Error {response.status_code}: {response.text}")
                
        except Exception as e:
            print(f"[TELEGRAM] Excepción: {e}")
