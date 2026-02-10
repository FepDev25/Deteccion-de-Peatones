# Sistema Hibrido de Deteccion de Peatones con Analisis de Postura

## Descripcion del Proyecto

Este repositorio contiene la implementacion de un sistema de vision por computador para la deteccion de peatones en tiempo real con analisis de postura corporal. El sistema combina tecnicas clasicas de vision por computador (HOG y LBP) con aprendizaje profundo (YOLOv8-pose) mediante una arquitectura hibrida de tres niveles que incluye una aplicacion de escritorio en C++/Qt, un servidor Flask en Python, y un bot de Telegram para notificaciones.

El sistema esta disenado para detectar personas en diferentes posturas (de pie, agachadas, sentadas, inclinadas) y reducir significativamente los falsos positivos mediante un pipeline de validacion de seis filtros heuristicos.

## Arquitectura del Sistema

El sistema implementa una arquitectura de tres niveles:

### Nivel 1: Aplicacion de Escritorio (C++/Qt + OpenCV)
- Captura de video en tiempo real desde webcam (640x480 @ 30 FPS)
- Detector hibrido HOG (pre-entrenado) + LBP (entrenado especificamente para el proyecto)
- Validacion heuristica con 6 filtros para eliminar falsos positivos
- Envio via HTTP POST al servidor Flask cuando se detectan personas
- Interfaz grafica con visualizacion de FPS, conteo de detecciones y controles

### Nivel 2: Servidor Flask (Python)
- Recepcion de imagenes via HTTP POST desde la aplicacion C++
- Cola de procesamiento (max 50 elementos) con worker thread dedicado
- Inferencia con YOLOv8-pose (modelo yolov8n-pose.pt, 17 keypoints COCO)
- Generacion de imagen anotada con esqueleto, bounding boxes y metricas
- Creacion de GIF animado (6 segundos, 200ms por frame) con buffer de 30 frames

### Nivel 3: Bot de Telegram
- Recepcion de imagenes procesadas via Bot API
- Envio de tres salidas: imagen original, imagen anotada con esqueleto, y GIF animado
- Notificacion con metricas de rendimiento (FPS, memoria RAM, keypoints detectados, confianza promedio)

## Caracteristicas Principales

- **Deteccion multi-postura**: Capaz de detectar personas de pie, agachadas, sentadas e inclinadas
- **Reduccion de falsos positivos**: Pipeline de 6 filtros que reduce falsos positivos en un 83% comparado con HOG estandar
- **Procesamiento en tiempo real**: 26-33 FPS en CPU (solo deteccion) y 26 FPS con analisis completo en GPU
- **Analisis de postura**: Esqueleto de 17 keypoints COCO con confianza promedio del 86-96%
- **Notificaciones inteligentes**: Sistema de bot de Telegram con tres tipos de salida visual

## Estructura del Repositorio

```
.
├── app-cpp/                    # Aplicacion de escritorio C++/Qt
│   ├── src/                    # Codigo fuente C++
│   │   ├── main.cpp            # Punto de entrada Qt GUI
│   │   ├── mainwindow.cpp      # Logica principal del detector hibrido
│   │   ├── mainwindow.h        # Headers de la interfaz
│   │   ├── main_hybrid.cpp     # Version consola del detector
│   │   └── CMakeLists.txt      # Configuracion de compilacion
│   └── cascade_crouching.xml   # Clasificador LBP entrenado (posturas agachadas/sentadas)
│
├── bot-telegram/               # Servidor Flask y Bot de Telegram
│   ├── src/
│   │   ├── main.py             # Servidor Flask con cola de procesamiento
│   │   ├── detector.py         # Clase YOLOv8-pose y generacion de GIF
│   │   ├── bot.py              # Cliente Bot de Telegram
│   │   └── config.py           # Configuracion de variables de entorno
│   └── requirements.txt        # Dependencias Python
│
├── training/                   # Documentacion y recursos del entrenamiento LBP
│   ├── explicacion-training.md # Metodologia de entrenamiento del clasificador LBP
│   └── docker/                 # Configuracion Docker para entrenamiento
│
├── docs/                       # Documentacion del proyecto
│   ├── INFORME_IEEE.tex        # Paper en formato IEEEtran
│   └── report/
│       ├── graficas/           # Graficas de resultados experimentales
│       └── pruebas/            # Imagenes de pruebas del sistema
│
└── README.md                   # Este archivo
```

## Requisitos

### Para la aplicacion C++ (Nivel 1)
- Qt 5.15 o superior
- OpenCV 4.5+ con modulos imgproc, highgui, objdetect
- libcurl para envio HTTP
- CMake 3.16+
- Compilador C++17 (GCC 9+, Clang 10+, MSVC 2019+)

### Para el servidor Flask (Nivel 2)
- Python 3.8 o superior
- PyTorch 2.0+
- Ultralytics YOLOv8
- Flask 2.0+
- Pillow para generacion de GIF

### Para el Bot de Telegram (Nivel 3)
- python-telegram-bot 20+
- Token de Bot de Telegram (obtener via @BotFather)
- Chat ID del destinatario

## Instalacion y Ejecucion

### 1. Aplicacion C++ (Nivel 1)

```bash
cd app-cpp
mkdir build && cd build
cmake ..
make -j$(nproc)
./PedestrianDetector
```

La aplicacion iniciara la interfaz grafica con las siguientes opciones:
- Selector de camara (indice 0 por defecto)
- Modo de deteccion: HOG solo / LBP solo / Hibrido (recomendado)
- Checkbox para envio automatico al servidor Flask

### 2. Servidor Flask (Nivel 2)

```bash
cd bot-telegram
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
pip install -r requirements.txt

# Configurar variables de entorno
export TELEGRAM_TOKEN="tu_token_aqui"
export TELEGRAM_CHAT_ID="tu_chat_id_aqui"
export FLASK_HOST="0.0.0.0"
export FLASK_PORT="5000"

# Ejecutar servidor
python src/main.py
```

El servidor escuchara en `http://localhost:5000/detect` para recibir imagenes.

### 3. Bot de Telegram (Nivel 3)

El bot se inicializa automaticamente desde el servidor Flask. Para obtener las notificaciones:
1. Crear un bot nuevo via @BotFather en Telegram
2. Copiar el token proporcionado
3. Obtener el chat ID conversando con el bot @userinfobot
4. Configurar las variables de entorno correspondientes

## Detalles Tecnicos

### Entrenamiento del Clasificador LBP

El clasificador LBP (`cascade_crouching.xml`) fue entrenado especificamente para detectar personas en posturas no erguidas (agachadas, sentadas). Detalles del entrenamiento:

- **Dataset**: MPII Human Pose Dataset
- **Positivas**: 7,300 imagenes de personas agachadas/sentadas (relacion de aspecto 0.8-1.5)
- **Negativas**: 4,500 imagenes sin personas
- **Ventana de entrenamiento**: 32x32 pixeles
- **Etapas**: 15 (cascade)
- **Feature type**: LBP (Local Binary Patterns)
- **minHitRate**: 0.999
- **maxFalseAlarmRate**: 0.5

### Pipeline de Validacion (6 Filtros)

Cada region de interes candidata pasa por los siguientes filtros heuristicos:

| Filtro | Umbral | Proposito |
|--------|--------|-----------|
| Varianza de intensidad | > 17.0 | Descartar regiones uniformes |
| Densidad de bordes Canny | 0.08 - 0.40 | Requerir estructura de bordes |
| Gradientes Sobel | > 9.0 | Detectar gradientes verticales tipicos de personas |
| Transiciones LBP | > 0.15 | Medir textura de la region |
| Relacion de aspecto | < 2.3 | Filtrar objetos horizontales |
| Color de piel HSV | > 1.2% | Presencia de tonos de piel |

### Parametros Optimizados

**HOG (personas de pie):**
- `finalThreshold`: 1.8 (determinado tras 31 pruebas experimentales)
- `winStride`: 4x4
- `padding`: 8x8
- `scale`: 1.05
- Aspect ratio valido: 1.5 - 3.0

**LBP (personas agachadas/sentadas):**
- `scaleFactor`: 1.05
- `minNeighbors`: 12 (modo Qt) / 18 (modo consola)
- `minSize`: 60x50
- `maxSize`: 320x300
- Expansion de ROI: +25% ancho, +20% arriba, +60% abajo

## Resultados Experimentales

El sistema fue evaluado en 10 escenarios de prueba diferentes:

- **Precision promedio**: 85.8% AP
- **FPS (deteccion)**: 33 FPS en CPU / 70 FPS en GPU
- **FPS (analisis completo)**: 26 FPS en GPU
- **Reduccion de falsos positivos**: 83% comparado con HOG estandar
- **Keypoints detectados**: 15-17 puntos por persona (esqueleto COCO completo)
- **Confianza YOLOv8**: 86% - 96% en todas las pruebas

## Limitaciones Conocidas

- **Oclusiones severas**: La precision cae a 65% cuando mas del 50% del cuerpo esta oculto
- **Iluminacion extrema**: Bajo rendimiento con contraluz o sombras fuertes
- **Multi-persona densa**: Degradacion con mas de 5 personas superpuestas
- **Posturas inusuales**: Posible clasificacion erronea en posturas de yoga o gimnasia

## Trabajo Futuro

- Integracion de modelos de seguimiento (SORT, DeepSORT) para trayectorias temporales
- Implementacion de re-identificacion de personas entre frames
- Extension a video 4K con procesamiento piramidal multi-resolucion
- Exploracion de arquitecturas de atencion para mejorar deteccion con oclusiones
- Despliegue en hardware embebido (Jetson Xavier, Coral Edge TPU)

## Documentacion Adicional

- Paper academico en formato IEEE: `docs/INFORME_IEEE.tex`
- Metodologia de entrenamiento LBP: `training/explicacion-training.md`
- Imagenes de prueba y resultados: `docs/report/`

## Autores

- Diego Felipe Peralta Peralta (dperaltap2@est.ups.edu.ec)
- Samantha Micaela Suquilanda Quilli (ssuquilanda@est.ups.edu.ec)

Universidad Politecnica Salesiana, Cuenca - Ecuador

## Licencia

Este proyecto es de caracter academico. El codigo fuente, datasets y modelos entrenados estan disponibles para fines educativos y de investigacion.

## Referencias

- Dalal, N., & Triggs, B. (2005). Histograms of oriented gradients for human detection. CVPR 2005.
- Redmon, J., et al. (2016). You only look once: Unified, real-time object detection. CVPR 2016.
- Cao, Z., et al. (2017). Realtime multi-person 2D pose estimation using part affinity fields. CVPR 2017.
- Ultralytics. (2023). YOLOv8: A new state-of-the-art computer vision model.
