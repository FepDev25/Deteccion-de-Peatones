# Metodología de Entrenamiento del Clasificador LBP"

#### A. Adquisición y Preprocesamiento del Dataset

Para el entrenamiento del clasificador LBP especializado en posturas no erguidas (agachado/sentado), se utilizó el **MPII Human Pose Dataset**, dada su riqueza en actividades humanas variadas. El procesamiento de los datos se automatizó mediante scripts de Python (`Jupyter Notebooks`) para segregar las muestras en positivas y negativas:

1. **Extracción de Muestras Positivas:** Se desarrolló un algoritmo de extracción que itera sobre las anotaciones del dataset. Para cada sujeto anotado, se calculó la relación de aspecto () del *bounding box*.
* Se filtraron y recortaron las imágenes donde , correspondientes a geometrías cuadradas o casi cuadradas, típicas de personas agachadas o sentadas.
* Las muestras con  (verticales) fueron segregadas para otros propósitos, asegurando que el modelo LBP se especialice únicamente en morfologías complejas.
* Se obtuvieron aproximadamente 7,300 muestras positivas complejas.


2. **Generación de Muestras Negativas:** Para el fondo, se procesó el mismo dataset excluyendo cualquier imagen que contuviera identificadores de categoría "persona". Esto garantizó un conjunto de 4,500 imágenes de "fondo puro" (calles, interiores, paisajes) sin presencia humana, crucial para reducir la tasa de falsos positivos.

#### B. Entorno de Entrenamiento Virtualizado (Docker)

Debido a que las herramientas de entrenamiento en cascada (`opencv_traincascade`) fueron deprecadas y eliminadas en OpenCV 4.x, se implementó un entorno de entrenamiento aislado utilizando **Docker**.

Se construyó una imagen basada en **Ubuntu 18.04**, instalando explícitamente `libopencv-dev` y `opencv-data` desde los repositorios oficiales de dicha distribución. Esto permitió acceder a la versión estable de OpenCV 3.x y sus utilidades de C++ sin generar conflictos con las dependencias del sistema anfitrión. El contenedor se configuró para montar el volumen de trabajo local, permitiendo la persistencia de los modelos entrenados.

#### C. Generación de Vectores y Listas

Previo al entrenamiento, se estandarizó la entrada de datos mediante scripts auxiliares:

1. **Listas de Descripción:** Se ejecutó el script `Notess_crouching.py` para generar:
* `bg.txt`: Lista de rutas relativas a las 4,500 imágenes negativas.
* `info.txt`: Archivo de descripción de positivas que especifica la ruta, cantidad de objetos y coordenadas normalizadas () para cada recorte.


2. **Creación del Vector:** Se utilizó la utilidad `opencv_createsamples` para convertir las imágenes listadas en `info.txt` a un archivo binario (`positives.vec`), redimensionando todas las muestras a una geometría fija de  píxeles, adecuada para la detección de texturas LBP en posturas compactas.

#### D. Entrenamiento en Cascada (Cascade Training)

El entrenamiento del modelo se ejecutó dentro del contenedor Docker utilizando el algoritmo **AdaBoost** con características **LBP (Local Binary Patterns)**. A diferencia de las características Haar, LBP utiliza aritmética de enteros, lo que acelera significativamente tanto el entrenamiento como la inferencia en tiempo real.

El comando de entrenamiento final fue configurado con los siguientes hiperparámetros:

```bash
docker run --rm -v "$(pwd):/work" cv3-trainer \
    opencv_traincascade \
      -data data \
      -vec positives.vec \
      -bg bg.txt \
      -numPos 6500 \
      -numNeg 4500 \
      -numStages 15 \
      -w 32 -h 32 \
      -featureType LBP \
      -minHitRate 0.999 \
      -maxFalseAlarmRate 0.5

```

**Justificación de Parámetros:**

* **-numPos 6500 / -numNeg 4500:** Se utilizó una proporción alta de muestras disponibles para maximizar la generalización, reservando un pequeño porcentaje para validación.
* **-w 32 -h 32:** Se definió una ventana de detección cuadrada (relación de aspecto 1:1), coherente con la morfología de personas agachadas filtrada en la etapa de preprocesamiento.
* **-numStages 15:** Se estableció una cascada de 15 etapas fuertes.
* **-minHitRate 0.999:** Se forzó una sensibilidad extremadamente alta por etapa () para asegurar que el modelo no descarte peatones en las primeras capas de la cascada, delegando la discriminación fina a las etapas posteriores.
