import cv2
import numpy as np
from ultralytics import YOLO
import io
from PIL import Image

class PoseDetector:
    def __init__(self, model_path='yolov8n-pose.pt', confidence=0.5):
        print("[DETECTOR] Cargando modelo YOLOv8-Pose...")
        self.model = YOLO(model_path)
        self.confidence = confidence
        self.frame_buffer = []  # Buffer para crear GIF/video
        self.max_frames = 10    # Número de frames para el GIF
        print("[DETECTOR] Modelo cargado.")
    
    def analyze_image(self, image_bytes):
        try:
            np_bytes = np.frombuffer(image_bytes, np.uint8)
            frame = cv2.imdecode(np_bytes, cv2.IMREAD_COLOR)
            
            if frame is None:
                return {"success": False, "error": "Invalid image"}
            
            # Ejecutar detección de postura
            results = self.model(frame, conf=self.confidence)
            
            # Analizar postura
            status, is_critical = self._analyze_pose(results)
            
            # Generar imagen con anotaciones (posturas superpuestas)
            annotated_frame = results[0].plot()
            
            # Convertir imagen original a bytes
            success_orig, encoded_orig = cv2.imencode('.jpg', frame)
            if not success_orig:
                return {"success": False, "error": "Error encoding original image"}
            
            # Convertir imagen anotada a bytes
            success_ann, encoded_ann = cv2.imencode('.jpg', annotated_frame)
            if not success_ann:
                return {"success": False, "error": "Error encoding annotated image"}
            
            # Agregar frame al buffer para GIF
            self.frame_buffer.append(annotated_frame.copy())
            if len(self.frame_buffer) > self.max_frames:
                self.frame_buffer.pop(0)
            
            # Generar GIF solo si hay suficientes frames
            gif_bytes = None
            if len(self.frame_buffer) >= 3:
                gif_bytes = self._create_gif()
            
            return {
                "success": True,
                "status": status,
                "is_critical": is_critical,
                "original_image": encoded_orig.tobytes(),
                "annotated_image": encoded_ann.tobytes(),
                "gif_video": gif_bytes,
                "person_count": len(results[0].boxes) if results else 0
            }
            
        except Exception as e:
            print(f"[DETECTOR] Error: {e}")
            return {"success": False, "error": str(e)}
    
    def _create_gif(self):
        """Crear GIF animado de los últimos frames con posturas"""
        try:
            frames_rgb = []
            for frame in self.frame_buffer:
                # Convertir BGR a RGB para PIL
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(frame_rgb)
                frames_rgb.append(pil_img)
            
            # Guardar GIF en bytes
            gif_buffer = io.BytesIO()
            frames_rgb[0].save(
                gif_buffer,
                format='GIF',
                save_all=True,
                append_images=frames_rgb[1:],
                duration=200,  # 200ms por frame
                loop=0
            )
            gif_buffer.seek(0)
            return gif_buffer.getvalue()
            
        except Exception as e:
            print(f"[DETECTOR] Error creando GIF: {e}")
            return None
    
    def _analyze_pose(self, results):
        """Analizar postura basado en keypoints de YOLOv8-pose"""
        status = "Sin personas detectadas"
        is_critical = False
        
        if results and len(results[0].boxes) > 0:
            num_people = len(results[0].boxes)
            status_list = []
            
            for i in range(num_people):
                box = results[0].boxes[i].xywh[0]
                w, h = float(box[2]), float(box[3])
                aspect_ratio = w / h
                
                # Analizar postura por aspect ratio del bounding box
                if aspect_ratio > 1.2:
                    person_status = "CAÍDA DETECTADA "
                    is_critical = True
                elif aspect_ratio > 0.8:
                    person_status = "Agachado/Sentado "
                else:
                    person_status = "De Pie 🚶"
                
                status_list.append(person_status)
            
            # Crear mensaje final
            if num_people == 1:
                status = f"1 persona: {status_list[0]}"
            else:
                status = f"{num_people} personas detectadas\n" + "\n".join(
                    [f"• Persona {i+1}: {s}" for i, s in enumerate(status_list)]
                )
        
        print(f"[DETECTOR] Estado: {status}")
        return status, is_critical
