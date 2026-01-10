import cv2
import numpy as np
from ultralytics import YOLO

class PoseDetector:
    def __init__(self, model_path='yolov8n-pose.pt', confidence=0.5):
        print("[DETECTOR] Cargando modelo YOLOv8-Pose...")
        self.model = YOLO(model_path)
        self.confidence = confidence
        print("[DETECTOR] Modelo cargado.")
    
    def analyze_image(self, image_bytes):
        try:
            np_bytes = np.frombuffer(image_bytes, np.uint8)
            frame = cv2.imdecode(np_bytes, cv2.IMREAD_COLOR)
            
            if frame is None:
                return {"success": False, "error": "Invalid image"}
            
            results = self.model(frame, conf=self.confidence)
            
            status, is_critical = self._analyze_pose(results)
            annotated_frame = results[0].plot()
            
            success, encoded_img = cv2.imencode('.jpg', annotated_frame)
            if not success:
                return {"success": False, "error": "Error encoding image"}
            
            return {
                "success": True,
                "status": status,
                "is_critical": is_critical,
                "annotated_image": encoded_img.tobytes()
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _analyze_pose(self, results):
        status = "Desconocido"
        is_critical = False
        
        if results and len(results[0].boxes) > 0:
            box = results[0].boxes[0].xywh[0]
            w, h = float(box[2]), float(box[3])
            
            aspect_ratio = w / h
            
            if aspect_ratio > 1.2:
                status = "CAIDA DETECTADA"
                is_critical = True
            elif aspect_ratio > 0.8:
                status = "Agachado o Sentado"
            else:
                status = "De Pie o Caminando"
        
        print(f"[DETECTOR] Estado: {status}")
        return status, is_critical
