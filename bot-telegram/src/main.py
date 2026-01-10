from flask import Flask, request, jsonify
from detector import PoseDetector
from telegram_service import TelegramService
from config import Config

app = Flask(__name__)

config = Config()
detector = PoseDetector()
telegram_service = TelegramService(config.telegram_token, config.telegram_chat_id)

@app.route('/detect', methods=['POST'])
def detect():
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image provided"}), 400

        file = request.files['image']
        image_bytes = file.read()
        
        result = detector.analyze_image(image_bytes)
        
        if result['success']:
            telegram_service.send_notification(
                result['annotated_image'], 
                result['status']
            )
            
            return jsonify({
                "status": "success", 
                "analysis": result['status'],
                "is_critical": result['is_critical']
            }), 200
        else:
            return jsonify({"error": result['error']}), 500

    except Exception as e:
        print(f"[ERROR] {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)