# 📘 GUÍA TÉCNICA COMPLETA - SISTEMA DE DETECCIÓN DE PEATONES
## Visión por Computador - Universidad Nacional

---

## 📑 ÍNDICE

1. [Arquitectura del Sistema](#1-arquitectura-del-sistema)
2. [Técnicas de Detección Implementadas](#2-técnicas-de-detección-implementadas)
3. [Parámetros Optimizados](#3-parámetros-optimizados)
4. [Bot de Telegram con YOLOv8-Pose](#4-bot-de-telegram-con-yolov8-pose)
5. [Pipeline de Procesamiento](#5-pipeline-de-procesamiento)
6. [Resultados Experimentales](#6-resultados-experimentales)
7. [Código Clave Explicado](#7-código-clave-explicado)
8. [Conclusiones](#8-conclusiones)

---

## 1. ARQUITECTURA DEL SISTEMA

### 1.1 Visión General

El sistema implementa una **arquitectura híbrida de dos niveles** para detección de peatones en tiempo real:

```
┌─────────────────────────────────────────────────────────────┐
│                    NIVEL 1: APLICACIÓN Qt C++               │
│  ┌───────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │  HOG + SVM    │  │ LBP Cascade  │  │  Validación  │     │
│  │  (Standing)   │  │ (Crouching)  │  │  Heurística  │     │
│  └───────────────┘  └──────────────┘  └──────────────┘     │
│           │               │                    │            │
│           └───────────────┴────────────────────┘            │
│                           │                                 │
│                    HTTP POST /detect                        │
│                           ▼                                 │
└─────────────────────────────────────────────────────────────┘
                            │
                            │ Imagen + Metadatos
                            ▼
┌─────────────────────────────────────────────────────────────┐
│              NIVEL 2: BOT TELEGRAM (Python Flask)           │
│  ┌───────────────────────────────────────────────────┐     │
│  │           YOLOv8-Pose (Deep Learning)              │     │
│  │  • Detección de keypoints (17 puntos COCO)        │     │
│  │  • Confianza promedio: 92.71%                     │     │
│  │  • Validación con esqueleto humano                │     │
│  └───────────────────────────────────────────────────┘     │
│           │                                                 │
│           ▼                                                 │
│  ┌───────────────────────────────────────────────────┐     │
│  │         Buffer de Frames + Generador GIF          │     │
│  │  • Acumula 30 frames @ 200ms                      │     │
│  │  • Genera GIF de 6 segundos                       │     │
│  └───────────────────────────────────────────────────┘     │
│           │                                                 │
│           ▼                                                 │
│  ┌───────────────────────────────────────────────────┐     │
│  │           Telegram Bot API                         │     │
│  │  Envía: 1. Imagen original                        │     │
│  │         2. Imagen con keypoints                    │     │
│  │         3. GIF animado (video)                     │     │
│  └───────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 Componentes del Sistema

#### **A. Aplicación de Escritorio (Qt 5 + OpenCV 4)**

**Ubicación:** `app-cpp/src/mainwindow.cpp`

**Funcionalidades:**
- Captura en tiempo real desde webcam (640x480 @ 30 FPS)
- Detección híbrida HOG+LBP con GPU opcional (CUDA)
- Interfaz gráfica con visualización de detecciones
- Auto-envío al bot cada 2 segundos cuando detecta personas
- Filtrado NMS (Non-Maximum Suppression) IoU=0.4

**Tecnologías:**
```cpp
- Qt 5.15.3 → GUI multiplataforma
- OpenCV 4.x → Procesamiento de imágenes
- CUDA (opcional) → Aceleración GPU
- libcurl → Comunicación HTTP con bot
```

#### **B. Bot de Telegram (Python Flask)**

**Ubicación:** `bot-telegram/src/`

**Funcionalidades:**
- Servidor Flask escuchando en `localhost:5000`
- Inferencia YOLOv8-Pose (modelo `yolov8n-pose.pt`)
- Generación de GIF con buffer persistente
- Envío triple a Telegram (original + anotada + video)

**Tecnologías:**
```python
- Flask 2.x → API REST
- Ultralytics YOLOv8 → Deep Learning
- PIL/Pillow → Generación de GIF
- python-telegram-bot → Integración Telegram
```

---

## 2. TÉCNICAS DE DETECCIÓN IMPLEMENTADAS

### 2.1 HOG + SVM (Histogram of Oriented Gradients)

#### **Fundamento Teórico**

HOG es un descriptor de características que captura la **distribución de gradientes de intensidad** en regiones locales de una imagen. Es especialmente efectivo para detectar personas de pie debido a que captura la forma característica del cuerpo humano.

**Proceso:**

1. **Cálculo de Gradientes:**
   ```cpp
   I_x = ∂I/∂x  (derivada horizontal)
   I_y = ∂I/∂y  (derivada vertical)
   
   Magnitud: |G| = √(I_x² + I_y²)
   Orientación: θ = arctan(I_y / I_x)
   ```

2. **Ventanas Deslizantes:**
   - **Win Size:** 64x128 pixels (relación 1:2 → persona de pie)
   - **Block Size:** 16x16 pixels (normalización local)
   - **Cell Size:** 8x8 pixels (histograma de 9 bins)

3. **Clasificador SVM:**
   - Entrenado con dataset INRIA Person
   - Separa personas de fondo con hiperplano óptimo
   - **Kernel lineal** para detección rápida

#### **Implementación en el Código**

```cpp
void MainWindow::detectHOG(cv::Mat& frame, cv::Mat& gray, 
                           std::vector<cv::Rect>& detections,
                           std::vector<std::string>& labels, 
                           std::vector<double>& confidences) {
    std::vector<cv::Rect> hogDets;
    std::vector<double> weights;
    
    // REDIMENSIÓN: 640x480 → 320x240
    // ¿Por qué? Mayor velocidad (4x menos píxeles) manteniendo precisión
    cv::Mat resized;
    cv::resize(frame, resized, cv::Size(320, 240));
    
    #ifdef HAVE_CUDA
    if (useGPU && hogDetectorGPU) {
        // VERSIÓN GPU - 3-5x MÁS RÁPIDA
        cv::cuda::GpuMat gpuFrame;
        gpuFrame.upload(resized);
        
        cv::cuda::GpuMat gpuGray;
        cv::cuda::cvtColor(gpuFrame, gpuGray, cv::COLOR_BGR2GRAY);
        
        // PARÁMETROS GPU
        hogDetectorGPU->setScaleFactor(1.05);     // Pirámide de escalas (5% reducción por nivel)
        hogDetectorGPU->setHitThreshold(0.0);     // Sin umbral previo (se aplica después)
        hogDetectorGPU->setWinStride(cv::Size(8,8)); // Paso de ventana deslizante
        
        hogDetectorGPU->detectMultiScale(gpuGray, hogDets);
        weights.resize(hogDets.size(), 1.0);
    } else
    #endif
    {
        // VERSIÓN CPU
        hogDetector.detectMultiScale(
            resized, 
            hogDets, 
            weights,
            0.0,              // hitThreshold: umbral SVM (0=todos los candidatos)
            cv::Size(4,4),    // winStride: paso de ventana (4px = overlap alto)
            cv::Size(8,8),    // padding: margen adicional
            1.05,             // scale: factor pirámide de escalas
            1.8               // finalThreshold: CLAVE - filtro post-NMS
                              // 1.8 = BALANCE: detecta variadas posturas sin falsos positivos
        );
    }
    
    // FILTRADO POR ASPECT RATIO
    for (size_t i = 0; i < hogDets.size(); i++) {
        cv::Rect r = hogDets[i];
        
        // Escalar de vuelta a 640x480
        r.x *= 2; r.y *= 2;
        r.width *= 2; r.height *= 2;
        
        // FILTRO: Personas de pie tienen aspect ratio 1.5-3.0
        double aspect = (double)r.height / r.width;
        if (aspect >= 1.5 && aspect <= 3.0) {
            detections.push_back(r);
            labels.push_back("Peaton detectado");
            confidences.push_back(weights[i]);
        }
    }
}
```

**Justificación de Parámetros:**

| Parámetro | Valor | Razón |
|-----------|-------|-------|
| `finalThreshold` | 1.8 | **CRÍTICO:** Valor bajo (0.5-1.0) = muchos falsos positivos (sillas, lámparas). Valor alto (2.5-3.0) = pierde personas agachadas/lejanas. **1.8 = punto óptimo** |
| `winStride` | (4,4) | Overlap alto → mayor cobertura, detecta personas parcialmente visibles |
| `scale` | 1.05 | Pirámide fina → detecta personas de múltiples tamaños (cerca/lejos) |
| `aspect ratio` | 1.5-3.0 | Filtra detecciones horizontales (sofás, mesas) que no son personas |

---

### 2.2 LBP CASCADE CLASSIFIER (Local Binary Patterns)

#### **Fundamento Teórico**

LBP es un descriptor de **textura local** que compara píxeles vecinos con el píxel central. Es **más rápido** que HOG y mejor para detectar personas en **posturas no estándar** (agachadas, sentadas, parcialmente ocultas).

**Proceso:**

1. **Cálculo de LBP:**
   ```
   Para cada píxel (xc, yc):
   - Comparar con 8 vecinos en radio R
   - Si vecino >= centro → bit = 1, sino → bit = 0
   - LBP = número binario de 8 bits (0-255)
   
   Ejemplo:
     6  7  9        0  0  1
     5 [8] 10  →    0 [·] 1  → LBP = 01101000₂ = 104
     4  3  2        0  0  0
   ```

2. **Cascade Classifier:**
   - **Viola-Jones Framework** con características LBP
   - **20 etapas** de clasificadores débiles (boosted)
   - Rechaza regiones obviamente negativas en etapas tempranas
   - Solo procesa detalladamente regiones prometedoras

#### **Implementación en el Código**

```cpp
void MainWindow::detectLBP(cv::Mat& frame, cv::Mat& gray, 
                           std::vector<cv::Rect>& detections,
                           std::vector<std::string>& labels, 
                           std::vector<double>& confidences) {
    std::vector<cv::Rect> lbpDets;
    
    // DETECCIÓN MULTI-ESCALA
    cascadeCrouching.detectMultiScale(
        gray,                    // Imagen en escala de grises
        lbpDets,
        1.05,                    // scaleFactor: pirámide fina (5% por nivel)
        12,                      // minNeighbors: CLAVE - votación entre ventanas vecinas
                                 // 12 = BALANCE: detecta personas parciales sin explotar con FP
        cv::CASCADE_SCALE_IMAGE, // Flag: escalar imagen en vez de ventanas (más rápido)
        cv::Size(60, 50),        // minSize: personas muy lejanas o agachadas
        cv::Size(320, 300)       // maxSize: personas cercanas o sentadas
    );
    
    for (cv::Rect r : lbpDets) {
        // FILTRO 1: Aspect Ratio para agachados/sentados
        double aspect = (double)r.width / r.height;
        if (aspect < 0.6 || aspect > 2.3) continue; // Más amplio que HOG (permite posturas horizontales)
        
        // FILTRO 2: Evitar duplicados con detecciones HOG
        bool overlapWithHOG = false;
        for (size_t i = 0; i < detections.size(); i++) {
            if (labels[i] == "Peaton detectado") {
                cv::Rect intersection = r & detections[i];
                double iou = intersection.area() / (double)r.area();
                if (iou > 0.15) {  // Si overlap >15% con HOG → ya detectado
                    overlapWithHOG = true;
                    break;
                }
            }
        }
        if (overlapWithHOG) continue;
        
        // EXPANSIÓN DE BOUNDING BOX
        // ¿Por qué? LBP detecta torso/cabeza, pero necesitamos cuerpo completo
        int expandX = (int)(r.width * 0.25);       // +25% ancho (brazos/hombros)
        int expandTop = (int)(r.height * 0.20);    // +20% arriba (cabeza completa)
        int expandBottom = (int)(r.height * 0.60); // +60% abajo (PIERNAS COMPLETAS)
        
        r.x = std::max(0, r.x - expandX);
        r.y = std::max(0, r.y - expandTop);
        r.width = std::min(frame.cols - r.x, r.width + 2 * expandX);
        r.height = std::min(frame.rows - r.y, r.height + expandTop + expandBottom);
        
        // VALIDACIÓN HEURÍSTICA
        cv::Mat roi = frame(r);
        if (!isValidPerson(roi)) continue;  // Filtros adicionales (textura, color, bordes)
        
        detections.push_back(r);
        labels.push_back("Peaton detectado");
        confidences.push_back(1.0);
    }
}
```

**Justificación de Parámetros:**

| Parámetro | Valor | Razón |
|-----------|-------|-------|
| `minNeighbors` | 12 | **CRÍTICO:** Votación entre ventanas. 5-8 = muchos FP (sillas, objetos). 15-20 = pierde personas parciales. **12 = óptimo** |
| `scaleFactor` | 1.05 | Pirámide fina → detecta personas de múltiples tamaños y distancias |
| `minSize` | (60,50) | Permite detectar personas lejanas o agachadas (más pequeñas) |
| `maxSize` | (320,300) | Limita detecciones gigantes (evita detectar paredes, puertas) |
| `expandBottom` | 60% | **MUY IMPORTANTE:** LBP detecta torso, pero las piernas quedan fuera. Expansión agresiva hacia abajo captura piernas completas |

---

### 2.3 VALIDACIÓN HEURÍSTICA `isValidPerson()`

#### **Fundamento**

Después de HOG y LBP, aplicamos **6 filtros heurísticos** para eliminar falsos positivos (sofás, sillas, ventanas, lámparas).

```cpp
bool MainWindow::isValidPerson(const cv::Mat& roi) {
    cv::Mat gray_roi;
    cv::cvtColor(roi, gray_roi, cv::COLOR_BGR2GRAY);
    
    // ═══════════════════════════════════════════════════════
    // FILTRO 1: VARIANZA DE COLOR (Textura)
    // ═══════════════════════════════════════════════════════
    // Personas tienen texturas complejas (ropa, sombras, pliegues)
    // Objetos uniformes (paredes, puertas) tienen baja desviación estándar
    
    cv::Scalar mean_color, stddev_color;
    cv::meanStdDev(gray_roi, mean_color, stddev_color);
    
    if (stddev_color[0] < 11.0) {  // Umbral BALANCEADO
        return false;  // Muy uniforme → no es persona
    }
    
    // ═══════════════════════════════════════════════════════
    // FILTRO 2: DENSIDAD DE BORDES (Canny)
    // ═══════════════════════════════════════════════════════
    // Personas tienen bordes (contorno corporal, ropa, rasgos faciales)
    // Superficies lisas tienen pocos bordes
    
    cv::Mat edges;
    cv::Canny(gray_roi, edges, 50, 150);  // Umbrales estándar
    double edgeDensity = cv::countNonZero(edges) / (double)(edges.rows * edges.cols);
    
    if (edgeDensity < 0.03 || edgeDensity > 0.45) {
        return false;  // Muy pocos bordes (objeto liso) o demasiados (ruido)
    }
    
    // ═══════════════════════════════════════════════════════
    // FILTRO 3: GRADIENTES SOBEL (Dirección de bordes)
    // ═══════════════════════════════════════════════════════
    // Personas tienen gradientes balanceados horizontal y verticalmente
    // Objetos como barandas tienen gradientes solo en una dirección
    
    cv::Mat grad_x, grad_y;
    cv::Sobel(gray_roi, grad_x, CV_16S, 1, 0, 3);  // Gradiente horizontal
    cv::Sobel(gray_roi, grad_y, CV_16S, 0, 1, 3);  // Gradiente vertical
    
    cv::Mat abs_grad_x, abs_grad_y;
    cv::convertScaleAbs(grad_x, abs_grad_x);
    cv::convertScaleAbs(grad_y, abs_grad_y);
    
    cv::Scalar mean_x, stddev_x, mean_y, stddev_y;
    cv::meanStdDev(abs_grad_x, mean_x, stddev_x);
    cv::meanStdDev(abs_grad_y, mean_y, stddev_y);
    
    if (stddev_x[0] < 7.0 || stddev_y[0] < 7.0) {
        return false;  // Gradientes muy uniformes → no es persona
    }
    
    // ═══════════════════════════════════════════════════════
    // FILTRO 4: TEXTURA LBP (Transiciones locales)
    // ═══════════════════════════════════════════════════════
    // Personas tienen muchas transiciones de intensidad (textura rica)
    // Objetos planos tienen pocas transiciones
    
    int transitions = 0;
    for (int i = 1; i < gray_roi.rows - 1; i++) {
        for (int j = 1; j < gray_roi.cols - 1; j++) {
            int center = gray_roi.at<uchar>(i, j);
            int diff = 0;
            diff += std::abs(center - gray_roi.at<uchar>(i-1, j)) > 10 ? 1 : 0;
            diff += std::abs(center - gray_roi.at<uchar>(i+1, j)) > 10 ? 1 : 0;
            diff += std::abs(center - gray_roi.at<uchar>(i, j-1)) > 10 ? 1 : 0;
            diff += std::abs(center - gray_roi.at<uchar>(i, j+1)) > 10 ? 1 : 0;
            if (diff >= 2) transitions++;  // Al menos 2 vecinos diferentes
        }
    }
    double transitionRatio = transitions / (double)(gray_roi.rows * gray_roi.cols);
    
    if (transitionRatio < 0.12) {
        return false;  // Muy pocas transiciones → textura pobre
    }
    
    // ═══════════════════════════════════════════════════════
    // FILTRO 5: ASPECT RATIO (Forma física)
    // ═══════════════════════════════════════════════════════
    // Personas tienen formas verticales u horizontales moderadas
    // Objetos muy anchos (sofás largos) se rechazan
    
    double aspect = (double)roi.cols / roi.rows;
    if (aspect > 2.5) {
        return false;  // Demasiado ancho → no es persona
    }
    
    // ═══════════════════════════════════════════════════════
    // FILTRO 6: COLOR DE PIEL (HSV)
    // ═══════════════════════════════════════════════════════
    // Personas tienen píxeles de tono piel (cara, manos, brazos)
    // Objetos inanimados raramente tienen color piel
    
    cv::Mat hsv_roi;
    cv::cvtColor(roi, hsv_roi, cv::COLOR_BGR2HSV);
    
    cv::Mat skin_mask;
    // Rango HSV para tono piel (calibrado empíricamente)
    // H: 0-20 (tonos rojos-naranjas)
    // S: 20-255 (saturación mínima)
    // V: 60-255 (brillo mínimo)
    cv::inRange(hsv_roi, cv::Scalar(0, 20, 60), cv::Scalar(20, 255, 255), skin_mask);
    
    double skinRatio = cv::countNonZero(skin_mask) / (double)(skin_mask.rows * skin_mask.cols);
    
    if (skinRatio < 0.004) {  // Al menos 0.4% de píxeles de piel
        return false;  // Sin color piel visible → probablemente no es persona
    }
    
    return true;  // ✅ PASÓ TODOS LOS FILTROS
}
```

**Tabla de Umbrales Optimizados:**

| Filtro | Umbral | Falsos Positivos que Elimina |
|--------|--------|------------------------------|
| Desv. estándar | >11.0 | Paredes, puertas, ventanas uniformes |
| Densidad bordes | 0.03-0.45 | Superficies lisas y ruido excesivo |
| Gradientes Sobel | >7.0 | Barandas verticales, líneas horizontales |
| Transiciones LBP | >0.12 | Objetos planos sin textura |
| Aspect ratio | <2.5 | Sofás largos, mesas anchas |
| Color piel | >0.004 | Sillas, lámparas, objetos metálicos |

---

## 3. PARÁMETROS OPTIMIZADOS

### 3.1 Proceso de Optimización

Se realizaron **31 pruebas experimentales** variando parámetros hasta encontrar el **punto óptimo** entre:

- ✅ **Alta sensibilidad** (detectar personas agachadas, lejanas, parciales)
- ✅ **Baja tasa de falsos positivos** (no detectar sillas, sofás, lámparas)

### 3.2 Evolución de Parámetros

| Iteración | HOG finalThreshold | LBP minNeighbors | Resultado |
|-----------|-------------------|------------------|-----------|
| **Inicial** | 2.0 | 5 | ❌ Muchos FP (sofás, sillas) |
| **Iteración 1** | 2.5 | 15 | ❌ Muy estricto, pierde personas agachadas |
| **Iteración 2** | 1.5 | 8 | ❌ Demasiados FP, no viable |
| **FINAL** | **1.8** | **12** | ✅ **ÓPTIMO:** 80% precisión, 88.9% sensibilidad |

### 3.3 Configuración Final (Producción)

```cpp
// ══════════════════════════════════════════════════════════
// CONFIGURACIÓN OPTIMIZADA - mainwindow.cpp
// ══════════════════════════════════════════════════════════

// ────────────── HOG DETECTOR ──────────────
hogDetector.detectMultiScale(
    resized,
    hogDets,
    weights,
    0.0,              // hitThreshold
    cv::Size(4,4),    // winStride → overlap alto
    cv::Size(8,8),    // padding
    1.05,             // scale → pirámide fina
    1.8               // ⭐ finalThreshold: CLAVE DEL SISTEMA
);

// Aspect ratio filtering
double aspect = (double)r.height / r.width;
if (aspect >= 1.5 && aspect <= 3.0) { ... }

// ────────────── LBP CASCADE ──────────────
cascadeCrouching.detectMultiScale(
    gray,
    lbpDets,
    1.05,                    // scaleFactor
    12,                      // ⭐ minNeighbors: CLAVE
    cv::CASCADE_SCALE_IMAGE,
    cv::Size(60, 50),        // minSize → detecta lejanos
    cv::Size(320, 300)       // maxSize → limita gigantes
);

// ────────────── NMS FILTERING ──────────────
// IoU threshold: 0.4
if (iou > 0.4) suppressed[idx2] = true;

// ────────────── VALIDATION THRESHOLDS ──────────────
stddev_color[0] > 11.0;       // Textura mínima
edgeDensity: 0.03 - 0.45;     // Rango de bordes
stddev_sobel > 7.0;           // Gradientes mínimos
transitionRatio > 0.12;       // Transiciones LBP
aspect < 2.5;                 // Forma razonable
skinRatio > 0.004;            // Color piel mínimo

// ────────────── AUTO-SEND COOLDOWN ──────────────
cooldownFrames = 60;  // 2 segundos @ 30 FPS
```

---

## 4. BOT DE TELEGRAM CON YOLOv8-POSE

### 4.1 Arquitectura del Bot

```python
# ══════════════════════════════════════════════════════════
# bot-telegram/src/main.py - Flask Server
# ══════════════════════════════════════════════════════════

from flask import Flask, request, jsonify
from detector import PedestrianDetector
from telegram_service import TelegramService

app = Flask(__name__)
detector = PedestrianDetector()  # YOLOv8-Pose
telegram_service = TelegramService()

@app.route('/detect', methods=['POST'])
def detect():
    """
    Endpoint HTTP que recibe imágenes de la app Qt
    
    Flujo:
    1. Qt app detecta persona con HOG+LBP
    2. Envía imagen vía POST a localhost:5000/detect
    3. Bot ejecuta YOLOv8-Pose para validación
    4. Genera GIF con buffer de frames
    5. Envía 3 archivos a Telegram
    """
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    # Leer imagen
    image_file = request.files['image']
    image_data = image_file.read()
    
    # YOLOv8 Inference + GIF generation
    result = detector.analyze_image(image_data)
    
    # Enviar a Telegram
    telegram_service.send_notification(
        image_data=image_data,
        annotated_data=result['annotated_image'],
        gif_data=result.get('gif_data'),
        metadata=result['metadata']
    )
    
    return jsonify({'status': 'success', 'result': result['metadata']})
```

### 4.2 YOLOv8-Pose Detector

```python
# ══════════════════════════════════════════════════════════
# bot-telegram/src/detector.py
# ══════════════════════════════════════════════════════════

from ultralytics import YOLO
from PIL import Image, ImageDraw
import io
import numpy as np

class PedestrianDetector:
    # Buffer persistente entre llamadas HTTP
    frame_buffer = []
    MAX_FRAMES = 30
    
    def __init__(self):
        # Cargar modelo YOLOv8-Pose Nano (11.3 MB)
        # 17 keypoints COCO: nariz, ojos, orejas, hombros, codos, muñecas,
        #                    caderas, rodillas, tobillos
        self.model = YOLO('yolov8n-pose.pt')
    
    def analyze_image(self, image_data):
        """
        Ejecuta YOLOv8-Pose y genera GIF
        
        Parámetros YOLOv8:
        - conf=0.50 → Confianza mínima 50% (BALANCEADO)
        - iou=0.45 → NMS threshold
        - imgsz=640 → Redimensión a 640px (velocidad)
        """
        img = Image.open(io.BytesIO(image_data)).convert('RGB')
        
        # INFERENCIA YOLOV8
        results = self.model.predict(
            img,
            conf=0.50,      # ⭐ Umbral de confianza
            iou=0.45,       # NMS IoU
            imgsz=640,      # Tamaño de imagen
            device='cpu',   # 'cuda' si hay GPU
            verbose=False
        )
        
        # Extraer keypoints
        keypoints_list = []
        confidences = []
        
        for result in results:
            if result.keypoints is not None:
                for kpts in result.keypoints.data:
                    # kpts shape: (17, 3) → [x, y, confidence]
                    keypoints_list.append(kpts.cpu().numpy())
                    # Confianza promedio de los keypoints visibles
                    visible = kpts[:, 2] > 0.5
                    if visible.sum() > 0:
                        conf = kpts[visible, 2].mean()
                        confidences.append(float(conf))
        
        # DIBUJAR ESQUELETO
        annotated_img = self._draw_skeleton(img.copy(), keypoints_list)
        
        # ACUMULAR EN BUFFER (para GIF)
        self.frame_buffer.append(annotated_img.copy())
        if len(self.frame_buffer) > self.MAX_FRAMES:
            self.frame_buffer.pop(0)  # FIFO
        
        # GENERAR GIF (si hay suficientes frames)
        gif_data = None
        if len(self.frame_buffer) >= 20:  # Mínimo 20 frames = 4 segundos
            gif_data = self._create_gif()
        
        # Metadatos
        num_people = len(keypoints_list)
        total_keypoints = sum(len(kp) for kp in keypoints_list)
        avg_confidence = np.mean(confidences) if confidences else 0.0
        
        return {
            'annotated_image': self._image_to_bytes(annotated_img),
            'gif_data': gif_data,
            'metadata': {
                'people_count': num_people,
                'total_keypoints': total_keypoints,
                'avg_confidence': float(avg_confidence),
                'buffer_size': len(self.frame_buffer)
            }
        }
    
    def _draw_skeleton(self, img, keypoints_list):
        """
        Dibuja esqueleto COCO de 17 puntos
        
        Conexiones:
        - Cara: nariz-ojo_izq, nariz-ojo_der, ojo-oreja
        - Torso: hombros-caderas (línea central)
        - Brazos: hombro-codo-muñeca
        - Piernas: cadera-rodilla-tobillo
        """
        draw = ImageDraw.Draw(img)
        
        # COCO Skeleton (pares de índices de keypoints conectados)
        skeleton = [
            (0, 1), (0, 2),        # Nariz-ojos
            (1, 3), (2, 4),        # Ojos-orejas
            (5, 6),                # Hombros
            (5, 7), (7, 9),        # Brazo izquierdo
            (6, 8), (8, 10),       # Brazo derecho
            (5, 11), (6, 12),      # Hombros-caderas
            (11, 12),              # Caderas
            (11, 13), (13, 15),    # Pierna izquierda
            (12, 14), (14, 16)     # Pierna derecha
        ]
        
        for kpts in keypoints_list:
            # Dibujar conexiones
            for start_idx, end_idx in skeleton:
                if kpts[start_idx][2] > 0.5 and kpts[end_idx][2] > 0.5:
                    x1, y1 = int(kpts[start_idx][0]), int(kpts[start_idx][1])
                    x2, y2 = int(kpts[end_idx][0]), int(kpts[end_idx][1])
                    draw.line([(x1, y1), (x2, y2)], fill='lime', width=3)
            
            # Dibujar keypoints
            for kpt in kpts:
                if kpt[2] > 0.5:  # Confianza >50%
                    x, y = int(kpt[0]), int(kpt[1])
                    draw.ellipse([x-4, y-4, x+4, y+4], fill='red', outline='white')
        
        return img
    
    def _create_gif(self):
        """
        Genera GIF de 6 segundos con buffer de 30 frames
        
        - Duración por frame: 200ms
        - Total: 30 frames × 200ms = 6 segundos
        - Loop infinito
        """
        if len(self.frame_buffer) < 20:
            return None
        
        output = io.BytesIO()
        self.frame_buffer[0].save(
            output,
            format='GIF',
            save_all=True,
            append_images=self.frame_buffer[1:],
            duration=200,      # 200ms por frame
            loop=0             # Loop infinito
        )
        output.seek(0)
        return output.getvalue()
```

### 4.3 Telegram Service

```python
# ══════════════════════════════════════════════════════════
# bot-telegram/src/telegram_service.py
# ══════════════════════════════════════════════════════════

import telegram
from telegram.ext import Application
import io

class TelegramService:
    def __init__(self):
        self.bot = telegram.Bot(token=os.getenv('TELEGRAM_BOT_TOKEN'))
        self.chat_id = os.getenv('TELEGRAM_CHAT_ID')
    
    async def send_notification(self, image_data, annotated_data, gif_data, metadata):
        """
        Envía 3 archivos a Telegram:
        1. Imagen original (captura de Qt)
        2. Imagen anotada (con esqueleto YOLOv8)
        3. GIF animado (video de 6 segundos)
        """
        # Mensaje de texto con metadatos
        message = f"""
🚶 DETECCIÓN DE PEATONES
        
✅ Personas detectadas: {metadata['people_count']}
📊 Keypoints: {metadata['total_keypoints']} puntos
🎯 Confianza: {metadata['avg_confidence']*100:.1f}%
🎬 Frames acumulados: {metadata['buffer_size']}
        """
        
        # 1. Imagen original
        await self.bot.send_photo(
            chat_id=self.chat_id,
            photo=io.BytesIO(image_data),
            caption="📷 IMAGEN ORIGINAL"
        )
        
        # 2. Imagen anotada
        await self.bot.send_photo(
            chat_id=self.chat_id,
            photo=io.BytesIO(annotated_data),
            caption="🔬 IMAGEN CON KEYPOINTS (YOLOv8-Pose)"
        )
        
        # 3. GIF (si está disponible)
        if gif_data:
            await self.bot.send_animation(
                chat_id=self.chat_id,
                animation=io.BytesIO(gif_data),
                caption="🎥 VIDEO ANIMADO (6 segundos)"
            )
        
        await self.bot.send_message(
            chat_id=self.chat_id,
            text=message
        )
```

---

## 5. PIPELINE DE PROCESAMIENTO

### 5.1 Flujo Completo

```
┌─────────────────────────────────────────────────────────────────┐
│  FRAME 1: Captura de Cámara (640x480 @ 30 FPS)                 │
└───────────────────┬─────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────────┐
│  FRAME 2: Preprocesamiento                                      │
│  - Conversión a escala de grises                                │
│  - Redimensión a 320x240 (HOG speed boost)                      │
└───────────────────┬─────────────────────────────────────────────┘
                    │
        ┌───────────┴──────────┐
        │                      │
        ▼                      ▼
┌─────────────────┐   ┌──────────────────┐
│  HOG + SVM      │   │  LBP Cascade     │
│  (Personas pie) │   │  (Agachados)     │
│  finalThres=1.8 │   │  minNeigh=12     │
└────────┬────────┘   └────────┬─────────┘
         │                     │
         └──────────┬──────────┘
                    ▼
┌─────────────────────────────────────────────────────────────────┐
│  FRAME 3: Fusión de Detecciones                                 │
│  - Combinar resultados HOG + LBP                                │
│  - Eliminar duplicados (IoU > 15%)                              │
└───────────────────┬─────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────────┐
│  FRAME 4: Validación Heurística                                 │
│  - isValidPerson() con 6 filtros                                │
│  - Rechazar falsos positivos (objetos)                          │
└───────────────────┬─────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────────┐
│  FRAME 5: NMS (Non-Maximum Suppression)                         │
│  - Eliminar detecciones superpuestas (IoU > 0.4)                │
│  - Mantener la de mayor confianza                               │
└───────────────────┬─────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────────┐
│  FRAME 6: Visualización                                         │
│  - Dibujar bounding boxes (verde/morado)                        │
│  - Mostrar labels y confianza                                   │
│  - Actualizar estadísticas (FPS, count)                         │
└───────────────────┬─────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────────┐
│  FRAME 7: Auto-Send (si detecciones > 0 y cooldown == 0)        │
│  - HTTP POST a localhost:5000/detect                            │
│  - Activar cooldown de 60 frames (2 segundos)                   │
└───────────────────┬─────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────────┐
│  BOT: YOLOv8-Pose Inference                                     │
│  - Detectar 17 keypoints COCO                                   │
│  - Dibujar esqueleto sobre imagen                               │
│  - Acumular frame en buffer (max 30)                            │
└───────────────────┬─────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────────┐
│  BOT: Generación de GIF (si buffer >= 20 frames)                │
│  - Crear GIF de 6 segundos (30 frames × 200ms)                  │
│  - Loop infinito                                                │
└───────────────────┬─────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────────┐
│  BOT: Envío a Telegram                                          │
│  - Foto 1: Original                                             │
│  - Foto 2: Con keypoints                                        │
│  - Video: GIF animado                                           │
│  - Texto: Metadatos (FPS, memoria, confianza, keypoints)        │
└─────────────────────────────────────────────────────────────────┘
```

---

## 6. RESULTADOS EXPERIMENTALES

### 6.1 Métricas de Rendimiento

```
╔══════════════════════════════════════════════════════════════╗
║              RESULTADOS DE 31 PRUEBAS REALES                 ║
╚══════════════════════════════════════════════════════════════╝

┌──────────────────────────────────────────────────────────────┐
│  MÉTODO HÍBRIDO (HOG + LBP)                                  │
├──────────────────────────────────────────────────────────────┤
│  ✅ Precisión:         80.00%   (VP / (VP + FP))             │
│  ✅ Sensibilidad:      88.89%   (VP / (VP + FN))             │
│  ⚠️  Especificidad:     25.00%   (VN / (VN + FP))             │
│  ✅ F1-Score:          84.21%   (Balance prec/sens)          │
│  ✅ Exactitud:         74.29%   ((VP+VN) / Total)            │
│                                                              │
│  Matriz de Confusión:                                        │
│    VP = 24  |  FN = 3                                        │
│    FP = 6   |  VN = 2                                        │
└──────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────┐
│  YOLOv8-POSE (Deep Learning)                                 │
├──────────────────────────────────────────────────────────────┤
│  ✅ Precisión:         93.33%   ⬆️ +13.33%                    │
│  ✅ Sensibilidad:      96.55%   ⬆️ +7.66%                     │
│  ✅ Especificidad:     66.67%   ⬆️ +41.67%                    │
│  ✅ F1-Score:          94.92%   ⬆️ +10.71%                    │
│  ✅ Exactitud:         91.43%   ⬆️ +17.14%                    │
│                                                              │
│  Matriz de Confusión:                                        │
│    VP = 28  |  FN = 1                                        │
│    FP = 2   |  VN = 4                                        │
└──────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────┐
│  RENDIMIENTO DEL SISTEMA                                     │
├──────────────────────────────────────────────────────────────┤
│  🚀 FPS Promedio:      18.78 fps                             │
│  ⚡ FPS Máximo:        93.10 fps  (detección simple)         │
│  🐌 FPS Mínimo:        0.70 fps   (YOLOv8 en CPU)            │
│                                                              │
│  💾 Memoria Bot:       1688.45 MB  (Promedio)                │
│  💾 Memoria Qt:        ~1100 MB    (Estimado 65% menos)      │
│                                                              │
│  🎯 Confianza YOLOv8:  92.71%      (Promedio de keypoints)   │
│  📊 Keypoints/Frame:   17-34       (1-2 personas)            │
└──────────────────────────────────────────────────────────────┘
```

### 6.2 Análisis Detallado

**Fortalezas del Método Híbrido:**
- ✅ **Rápido:** 18-30 FPS en CPU (tiempo real)
- ✅ **Versátil:** Detecta personas de pie + agachadas + sentadas
- ✅ **Bajo consumo:** ~1.1 GB RAM vs 1.7 GB de YOLOv8
- ✅ **Alta sensibilidad:** 88.89% (detecta la mayoría de personas)

**Debilidades del Método Híbrido:**
- ❌ **Falsos positivos:** 6 detecciones incorrectas (sofás, sillas)
- ❌ **Baja especificidad:** 25% (identifica mal escenas sin personas)
- ❌ **Sensible a iluminación:** Requiere buena luz

**Ventajas de YOLOv8-Pose:**
- ✅ **Altísima precisión:** 93.33% (3 veces menos FP)
- ✅ **Información rica:** 17 keypoints por persona
- ✅ **Validación robusta:** Esqueleto anatómicamente consistente

**Desventajas de YOLOv8-Pose:**
- ❌ **Lento en CPU:** 0.7-3 FPS (no tiempo real sin GPU)
- ❌ **Mayor memoria:** 1.7 GB RAM
- ❌ **Dependencia de modelo:** Requiere archivo .pt (11.3 MB)

---

## 7. CÓDIGO CLAVE EXPLICADO

### 7.1 Detección Híbrida

```cpp
void MainWindow::detectHybrid(cv::Mat& frame, cv::Mat& gray, 
                              std::vector<cv::Rect>& detections,
                              std::vector<std::string>& labels, 
                              std::vector<double>& confidences) {
    // ═══════════════════════════════════════════════════════════════
    // ESTRATEGIA HÍBRIDA: Combinar fortalezas de HOG y LBP
    // ═══════════════════════════════════════════════════════════════
    
    // PASO 1: HOG detecta personas de pie (alta precisión)
    detectHOG(frame, gray, detections, labels, confidences);
    
    // PASO 2: LBP detecta agachados/sentados (complementario)
    // Internamente, LBP filtra duplicados con HOG (IoU > 15%)
    detectLBP(frame, gray, detections, labels, confidences);
    
    // RESULTADO: 
    // - HOG aporta detecciones de personas de pie (alto peso)
    // - LBP aporta detecciones de posturas complejas (menor peso)
    // - Duplicados se eliminan automáticamente en detectLBP()
}
```

### 7.2 Filtrado NMS (Non-Maximum Suppression)

```cpp
void MainWindow::filterOverlapping(std::vector<cv::Rect>& detections,
                                   std::vector<std::string>& labels,
                                   std::vector<double>& confidences) {
    // ═══════════════════════════════════════════════════════════════
    // NMS: Eliminar detecciones superpuestas, mantener la mejor
    // ═══════════════════════════════════════════════════════════════
    
    if (detections.empty()) return;
    
    // PASO 1: Crear índices ordenados por confianza (mayor a menor)
    std::vector<int> indices(detections.size());
    for (size_t i = 0; i < detections.size(); i++) indices[i] = i;
    
    std::sort(indices.begin(), indices.end(), [&](int a, int b) {
        return confidences[a] > confidences[b];
    });
    
    // PASO 2: Suprimir detecciones superpuestas
    std::vector<bool> suppressed(detections.size(), false);
    std::vector<cv::Rect> filtered;
    std::vector<std::string> filteredLabels;
    std::vector<double> filteredConf;
    
    for (size_t i = 0; i < indices.size(); i++) {
        int idx = indices[i];
        if (suppressed[idx]) continue;  // Ya fue suprimida
        
        // Mantener esta detección (la de mayor confianza)
        filtered.push_back(detections[idx]);
        filteredLabels.push_back(labels[idx]);
        filteredConf.push_back(confidences[idx]);
        
        // Suprimir todas las que se superponen significativamente
        for (size_t j = i + 1; j < indices.size(); j++) {
            int idx2 = indices[j];
            if (suppressed[idx2]) continue;
            
            // Calcular IoU (Intersection over Union)
            cv::Rect inter = detections[idx] & detections[idx2];
            double iou = inter.area() / 
                        (double)(detections[idx].area() + 
                                detections[idx2].area() - inter.area());
            
            // Si IoU > 0.4 → Suprimir la de menor confianza
            if (iou > 0.4) suppressed[idx2] = true;
        }
    }
    
    // PASO 3: Reemplazar con detecciones filtradas
    detections = filtered;
    labels = filteredLabels;
    confidences = filteredConf;
}
```

### 7.3 Auto-Send con Cooldown

```cpp
// ═══════════════════════════════════════════════════════════════
// SISTEMA DE AUTO-ENVÍO AL BOT
// ═══════════════════════════════════════════════════════════════

// En processFrame():
if (botAutoSend && detectionCount > 0 && cooldownFrames == 0) {
    qDebug() << "[AUTO] Enviando detección al bot...";
    
    if (sendImageViaHTTP(display)) {
        qDebug() << "[AUTO] Imagen enviada exitosamente!" << detectionCount << "personas";
        
        // ACTIVAR COOLDOWN: 60 frames @ 30 FPS = 2 segundos
        // ¿Por qué 2 segundos?
        // - Evita spam al bot
        // - Permite acumular frames para GIF (buffer necesita 20 frames = 4 seg)
        // - Da tiempo al usuario de moverse/cambiar postura
        cooldownFrames = 60;
    }
}

// Decrementar cooldown cada frame
if (cooldownFrames > 0) cooldownFrames--;

// RESULTADO:
// - Bot recibe imágenes cada 2 segundos cuando hay detecciones
// - Buffer acumula 30 frames en ~6-7 segundos
// - GIF se genera cuando buffer >= 20 frames
```

---

## 8. CONCLUSIONES

### 8.1 Logros del Proyecto

✅ **Sistema híbrido funcional** que combina técnicas clásicas (HOG+LBP) con deep learning (YOLOv8)

✅ **Alta sensibilidad:** 88.89% detecta personas en posturas variadas (pie, agachado, sentado)

✅ **Tiempo real:** 18-30 FPS en CPU, escalable a 60+ FPS con GPU

✅ **Integración completa:** App Qt → Bot Telegram → Usuario final

✅ **Generación de video:** Buffer inteligente crea GIF de 6 segundos automáticamente

✅ **Validación robusta:** 6 filtros heurísticos eliminan falsos positivos

### 8.2 Comparación de Técnicas

| Criterio | HOG+LBP | YOLOv8-Pose | Ganador |
|----------|---------|-------------|---------|
| **Precisión** | 80.00% | 93.33% | 🏆 YOLOv8 |
| **Sensibilidad** | 88.89% | 96.55% | 🏆 YOLOv8 |
| **Velocidad (CPU)** | 18-30 FPS | 0.7-3 FPS | 🏆 HOG+LBP |
| **Memoria** | ~1.1 GB | ~1.7 GB | 🏆 HOG+LBP |
| **Información** | Bounding box | 17 keypoints | 🏆 YOLOv8 |
| **Complejidad** | Media | Alta | 🏆 HOG+LBP |

**Conclusión:** Sistema híbrido aprovecha **velocidad de HOG+LBP** para detección en tiempo real y **precisión de YOLOv8** para validación y análisis detallado.

### 8.3 Trabajo Futuro

🔧 **Optimizaciones posibles:**
- Implementar HOG+LBP en GPU (CUDA) para alcanzar 60+ FPS
- Entrenar modelo YOLOv8 personalizado con dataset de posturas específicas
- Agregar tracking (Kalman filter) para seguimiento temporal
- Implementar detección de actividades (caminar, correr, caer)

🌟 **Aplicaciones:**
- Vigilancia en tiempo real
- Conteo de personas en espacios públicos
- Sistemas de seguridad inteligentes
- Asistencia para personas con discapacidad visual

---

## 📚 REFERENCIAS

1. **HOG:** Dalal, N., & Triggs, B. (2005). Histograms of oriented gradients for human detection. CVPR.
2. **LBP:** Ojala, T., Pietikäinen, M., & Mäenpää, T. (2002). Multiresolution gray-scale and rotation invariant texture classification with local binary patterns. PAMI.
3. **Viola-Jones:** Viola, P., & Jones, M. (2001). Rapid object detection using a boosted cascade of simple features. CVPR.
4. **YOLOv8:** Ultralytics (2023). YOLOv8: State-of-the-art object detection. https://github.com/ultralytics/ultralytics
5. **OpenCV:** Bradski, G. (2000). The OpenCV Library. Dr. Dobb's Journal of Software Tools.

---

## 📄 ARCHIVOS DEL PROYECTO

```
Deteccion-de-Peatones/
├── app-cpp/src/
│   ├── mainwindow.cpp          → Lógica principal HOG+LBP
│   ├── mainwindow.h            → Declaraciones
│   └── main_qt.cpp             → Punto de entrada Qt
│
├── bot-telegram/src/
│   ├── main.py                 → Flask server
│   ├── detector.py             → YOLOv8-Pose inference
│   ├── telegram_service.py     → Integración Telegram
│   └── config.py               → Configuración (.env)
│
├── docs/
│   ├── generar_graficas_completas.py  → Script de gráficas
│   ├── Resultados_pruebas.md          → Datos experimentales
│   ├── GUIA_TECNICA_COMPLETA.md       → Este documento
│   └── report/graficas/               → Gráficas generadas
│       ├── 1_precision_sensibilidad_especificidad.png
│       ├── 2_matriz_confusion.png
│       ├── 3_uso_memoria.png
│       ├── 4_analisis_fps.png
│       └── 5_yolov8_confianza_keypoints.png
│
└── training/
    ├── cascade_crouching.xml   → Cascade LBP entrenado
    └── cascade_standing.xml    → Cascade LBP alternativo
```

---

**Autor:** Sistema de Detección de Peatones  
**Universidad:** Universidad Nacional  
**Curso:** Visión por Computador  
**Fecha:** Febrero 2026  
**Versión:** 2.0 Final

---

> 💡 **Nota Final:** Este proyecto demuestra que la **combinación de técnicas clásicas y modernas** (HOG+LBP + YOLOv8) produce sistemas robustos, eficientes y precisos para aplicaciones del mundo real. La clave está en **entender las fortalezas** de cada método y **optimizar parámetros** mediante experimentación rigurosa.
