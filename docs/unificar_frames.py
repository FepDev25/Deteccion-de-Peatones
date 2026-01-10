import os
import shutil
from pathlib import Path

# --- CONFIGURACIÓN DE RUTAS (Modo Linux) ---

# 1. Dónde están las carpetas de Felipe (01, 02, 03...)
# Usamos expanduser para que entienda el simbolo "~" (home)
RUTA_ORIGEN = os.path.expanduser("~/Descargas/raw_data/temp_frames")

# 2. Dónde las vamos a guardar (Tu carpeta del proyecto)
# Asumiendo que ejecutas este script desde la raiz "DETECCION-DE-PEATONES"
RUTA_DESTINO = "training/raw_pos"

def unificar_imagenes():
    # Crear carpeta destino si no existe
    if not os.path.exists(RUTA_DESTINO):
        os.makedirs(RUTA_DESTINO)
        print(f"✅ Carpeta creada: {RUTA_DESTINO}")
    else:
        print(f"📂 Usando carpeta existente: {RUTA_DESTINO}")

    contador = 1
    archivos_movidos = 0
    
    # Verificamos que exista el origen
    if not os.path.exists(RUTA_ORIGEN):
        print(f"❌ ERROR: No encuentro la carpeta de origen: {RUTA_ORIGEN}")
        print("Verifica que la ruta sea correcta.")
        return

    print(f"🚀 Iniciando unificación desde: {RUTA_ORIGEN}")

    # Recorrer todas las carpetas (01, 02, 03...)
    for root, dirs, files in os.walk(RUTA_ORIGEN):
        # Ordenamos los archivos para que sigan un orden lógico
        files.sort()
        
        for filename in files:
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                # Ruta completa actual del archivo
                ruta_original = os.path.join(root, filename)
                
                # Nuevo nombre: persona_00001.jpg, persona_00002.jpg...
                nuevo_nombre = f"persona_{contador:05d}.jpg"
                ruta_final = os.path.join(RUTA_DESTINO, nuevo_nombre)
                
                try:
                    # Copiamos el archivo
                    shutil.copy2(ruta_original, ruta_final)
                    
                    # Log visual cada 500 fotos
                    if contador % 500 == 0:
                        print(f"   -> Procesadas {contador} imágenes...")
                    
                    contador += 1
                    archivos_movidos += 1
                    
                except Exception as e:
                    print(f"⚠️ Error copiando {filename}: {e}")

    print("-" * 50)
    print(f"✅ ¡TERMINADO! Total de imágenes listas: {archivos_movidos}")
    print(f"📍 Ubicación: {os.path.abspath(RUTA_DESTINO)}")
    print("-" * 50)

if __name__ == "__main__":
    unificar_imagenes()