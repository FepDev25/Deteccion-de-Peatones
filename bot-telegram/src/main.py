from flask import Flask, request, jsonify
from detector import PoseDetector
from telegram_service import TelegramService
from config import Config
import queue
import threading
import time

app = Flask(__name__)

config = Config()
detector = PoseDetector() # cargar YOLO
telegram_service = TelegramService(config.telegram_token, config.telegram_chat_id)

job_queue = queue.Queue(maxsize=50)

def worker():
    """Saca fotos de la cola y las procesa una por una, sin bloquear al cliente C++."""
    print("[WORKER] Hilo de procesamiento iniciado y esperando imágenes...")
    
    while True:
        try:
            # esperar a que llegue una imagen a la cola
            image_bytes = job_queue.get()
            
            start_time = time.time()
            print(f"[WORKER] Procesando imagen... (En cola: {job_queue.qsize()})")

            result = detector.analyze_image(image_bytes)
            
            if result['success']:
                # Enviar 3 archivos según la rúbrica
                telegram_service.send_notification(
                    original_image=result['original_image'],
                    annotated_image=result['annotated_image'],
                    gif_video=result.get('gif_video'),  # Puede ser None si no hay suficientes frames
                    status_text=result['status'],
                    person_count=result.get('person_count', 0)
                )
                print(f"[WORKER] Notificación enviada. Estado: {result['status']}")
            else:
                print(f"[WORKER] Error en detección: {result.get('error')}")

            # Log de tiempo real de procesamiento
            elapsed = time.time() - start_time
            print(f"[WORKER] Tarea finalizada en {elapsed:.2f}s")

        except Exception as e:
            print(f"[WORKER] Error crítico: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            job_queue.task_done()

threading.Thread(target=worker, daemon=True).start()


@app.route('/detect', methods=['POST'])
def detect():
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image provided"}), 400

        file = request.files['image']
        
        image_bytes = file.read()
        
        try:
            job_queue.put_nowait(image_bytes)
            
            return jsonify({
                "status": "queued", 
                "message": "Imagen recibida. Procesando en segundo plano."
            }), 200
            
        except queue.Full:
            print("[ERROR] La cola está llena, descartando imagen.")
            return jsonify({"error": "Server overload (Queue full)"}), 503

    except Exception as e:
        print(f"[ERROR API] {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # threaded=True permite manejar múltiples peticiones HTTP a la vez
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)