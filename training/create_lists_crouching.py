import os
import cv2

POS_DIR = 'positives_complex' 
NEG_DIR = 'negatives'

print("Generando lista de negativas...")
with open('bg.txt', 'w') as f:
    for filename in os.listdir(NEG_DIR):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            f.write(f'{NEG_DIR}/{filename}\n')

print("Generando lista de positivas (Crouching)...")
with open('info.txt', 'w') as f:
    count = 0
    for filename in os.listdir(POS_DIR):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            path = f'{POS_DIR}/{filename}'
            img = cv2.imread(path)
            if img is not None:
                h, w = img.shape[:2]
                # Formato estándar
                f.write(f'{path} 1 0 0 {w} {h}\n')
                count += 1
    print(f"Total imágenes complejas: {count}")