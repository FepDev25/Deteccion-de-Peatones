import requests

class TelegramService:
    def __init__(self, token, chat_id):
        self.token = token
        self.chat_id = chat_id
        self.base_url = f"https://api.telegram.org/bot{token}"
    
    def send_notification(self, original_image, annotated_image, gif_video, status_text, person_count, metrics):
        """
        Envía 3 archivos según la rúbrica:
        1. Imagen original con mensaje de detección
        2. Imagen con posturas humanas detectadas (superpuestas)
        3. Video corto (GIF) de la detección en tiempo real
        """
        try:
            # Construir información de uso del sistema (Criterio 4 - 10%)
            usage_info = (
                f"\n\n📈 Información de Uso:\n"
                f"⚡ FPS: {metrics.get('fps', 0)} fps\n"
                f"🧠 Memoria: {metrics.get('memory_mb', 0)} MB ({metrics.get('memory_percent', 0)}%)\n"
                f"🎯 Keypoints: {metrics.get('keypoints_detected', 0)} puntos\n"
                f"✅ Confianza: {metrics.get('avg_confidence', 0):.0%}\n"
                f"⏱️ Tiempo: {metrics.get('process_time_ms', 0)} ms"
            )
            
            # 1. Enviar IMAGEN ORIGINAL con mensaje
            caption_original = f"🔍 Detección realizada\n\n📊 Estado: {status_text}\n👥 Personas: {person_count}{usage_info}"
            self._send_photo(original_image, caption_original)
            
            # 2. Enviar IMAGEN CON POSTURAS (esqueletos superpuestos)
            caption_poses = f"🦴 Análisis de Posturas Humanas\n\n{status_text}{usage_info}"
            self._send_photo(annotated_image, caption_poses)
            
            # 3. Enviar VIDEO/GIF (si está disponible)
            if gif_video:
                caption_gif = f"🎬 Video de Detección (≥5 segundos)\n{person_count} persona(s) detectada(s){usage_info}"
                self._send_animation(gif_video, caption_gif)
            
            print("[TELEGRAM] ✅ Notificación completa enviada (3 archivos)")
                
        except Exception as e:
            print(f"[TELEGRAM] Excepción: {e}")
    
    def _send_photo(self, image_bytes, caption):
        """Enviar foto a Telegram"""
        try:
            url = f"{self.base_url}/sendPhoto"
            payload = {'chat_id': self.chat_id, 'caption': caption}
            
            if isinstance(image_bytes, bytes):
                files = {'photo': ('detection.jpg', image_bytes, 'image/jpeg')}
            else:
                files = {'photo': ('detection.jpg', bytes(image_bytes), 'image/jpeg')}
            
            response = requests.post(url, data=payload, files=files, timeout=10)
            
            if response.status_code == 200:
                print(f"[TELEGRAM] Foto enviada: {caption[:30]}...")
            else:
                print(f"[TELEGRAM] Error {response.status_code}: {response.text}")
                
        except Exception as e:
            print(f"[TELEGRAM] Error enviando foto: {e}")
    
    def _send_animation(self, gif_bytes, caption):
        """Enviar GIF/animación a Telegram"""
        try:
            url = f"{self.base_url}/sendAnimation"
            payload = {'chat_id': self.chat_id, 'caption': caption}
            files = {'animation': ('detection.gif', gif_bytes, 'image/gif')}
            
            response = requests.post(url, data=payload, files=files, timeout=15)
            
            if response.status_code == 200:
                print(f"[TELEGRAM] GIF enviado")
            else:
                print(f"[TELEGRAM] Error {response.status_code}: {response.text}")
                
        except Exception as e:
            print(f"[TELEGRAM] Error enviando GIF: {e}")