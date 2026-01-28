import os
import cv2

POS_DIR = 'positives_standing'
NEG_DIR = 'negatives'

with open('bg.txt', 'w') as f:
    for filename in os.listdir(NEG_DIR):
        if filename.endswith('.jpg'): f.write(f'{NEG_DIR}/{filename}\n')

with open('info.txt', 'w') as f:
    for filename in os.listdir(POS_DIR):
        if filename.endswith('.jpg'):
            path = f'{POS_DIR}/{filename}'
            img = cv2.imread(path)
            if img is not None:
                h, w = img.shape[:2]
                f.write(f'{path} 1 0 0 {w} {h}\n')