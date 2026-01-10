import cv2
import os
from pathlib import Path


def extract_frames(video_path, output_folder, frame_interval=5, max_frames=None):
    # Extrae frames de un video.
  
    os.makedirs(output_folder, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        print(f"Error: No se pudo abrir el video {video_path}")
        return 0
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"Procesando: {video_path.name}")
    
    frame_count = 0
    extracted_count = 0
    video_name = video_path.stem  # Nombre del video sin extensión
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Extraer solo cada N frames
        if frame_count % frame_interval == 0:
            # Nombre del archivo de salida
            output_filename = f"{video_name}_frame_{frame_count:06d}.jpg"
            output_path = os.path.join(output_folder, output_filename)
            # Guardar el frame
            cv2.imwrite(output_path, frame)
            extracted_count += 1
            # Verificar si alcanzamos el máximo
            if max_frames and extracted_count >= max_frames:
                break
        
        frame_count += 1
    
    cap.release()
    print(f"Extraídos {extracted_count} frames de {frame_count} frames totales")
    return extracted_count


def process_videos(videos_folder, output_base_folder, category, frame_interval=5, max_frames_per_video=None):
    # Procesa todos los videos de una categoría.

    videos_path = Path(videos_folder)
    
    if not videos_path.exists():
        print(f"Error: La carpeta {videos_path} no existe")
        return
    
    # Crear carpeta de salida
    output_folder = Path(output_base_folder) / category
    output_folder.mkdir(parents=True, exist_ok=True)
    
    # Buscar todos los archivos de video
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.MP4', '.AVI', '.MOV', '.MKV']
    video_files = []
    for ext in video_extensions:
        video_files.extend(videos_path.glob(f'*{ext}'))
    
    if not video_files:
        print(f"No se encontraron videos en {videos_path}")
        return
    
    print(f"Procesando categoría: {category.upper()}")
    
    total_extracted = 0
    
    for i, video_file in enumerate(video_files, 1):
        print(f"\n[{i}/{len(video_files)}]")
        extracted = extract_frames(
            video_file,
            output_folder,
            frame_interval=frame_interval,
            max_frames=max_frames_per_video
        )
        total_extracted += extracted
    
    print(f"Total de frames extraídos para {category}: {total_extracted}")


def main():
    # Configuración
    script_dir = Path(__file__).parent
    videos_base = script_dir / "videos"
    output_base = script_dir / "raw_data"
    
    # Parámetros de extracción
    FRAME_INTERVAL = 5  # Extraer cada 5 frames
    MAX_FRAMES_PER_VIDEO = None 
    
    # Procesar videos positivos
    process_videos(
        videos_base / "positivos",
        output_base,
        "positivos",
        frame_interval=FRAME_INTERVAL,
        max_frames_per_video=MAX_FRAMES_PER_VIDEO
    )
    
    # Procesar videos negativos
    process_videos(
        videos_base / "negativos",
        output_base,
        "negativos",
        frame_interval=FRAME_INTERVAL,
        max_frames_per_video=MAX_FRAMES_PER_VIDEO
    )
    
    print("realizado")


if __name__ == "__main__":
    main()
