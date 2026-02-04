import cv2
import numpy as np
from ultralytics import YOLO
import io
from PIL import Image
import psutil
import time

class PoseDetector:
    def __init__(self, model_path='yolov8n-pose.pt', confidence=0.5):
        print("[DETECTOR] Cargando modelo YOLOv8-Pose...")
        self.model = YOLO(model_path)
        self.confidence = confidence
        self.frame_buffer = []  # Buffer para crear GIF/video
        self.max_frames = 30    # 30 frames × 200ms = 6 segundos
        self.last_process_time = 0
        self.fps = 0
        print("[DETECTOR] Modelo cargado.")
    
    def analyze_image(self, image_bytes):
        try:
            start_time = time.time()
            
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
            if len(self.frame_buffer) >= 5:  # Mínimo 5 frames
                gif_bytes = self._create_gif()
            
            # Calcular FPS y métricas de uso
            process_time = time.time() - start_time
            self.fps = 1.0 / process_time if process_time > 0 else 0
            
            # Obtener información de uso del sistema
            memory_percent = psutil.virtual_memory().percent
            memory_mb = psutil.Process().memory_info().rss / 1024 / 1024
            
            # Contar keypoints detectados
            total_keypoints = 0
            avg_confidence = 0
            if results and len(results[0].keypoints) > 0:
                keypoints = results[0].keypoints.data
                for person_kpts in keypoints:
                    # Contar keypoints con confianza > 0.5
                    visible_kpts = person_kpts[person_kpts[:, 2] > 0.5]
                    total_keypoints += len(visible_kpts)
                    if len(visible_kpts) > 0:
                        avg_confidence += visible_kpts[:, 2].mean()
                
                if len(keypoints) > 0:
                    avg_confidence /= len(keypoints)
            
            return {
                "success": True,
                "status": status,
                "is_critical": is_critical,
                "original_image": encoded_orig.tobytes(),
                "annotated_image": encoded_ann.tobytes(),
                "gif_video": gif_bytes,
                "person_count": len(results[0].boxes) if results else 0,
                "fps": round(self.fps, 1),
                "memory_percent": round(memory_percent, 1),
                "memory_mb": round(memory_mb, 1),
                "keypoints_detected": total_keypoints,
                "avg_confidence": round(float(avg_confidence), 2) if avg_confidence > 0 else 0.0,
                "process_time_ms": round(process_time * 1000, 1)
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
