cmake_minimum_required(VERSION 3.16)
project(DeteccionPeatones CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 1. CONFIGURACIÓN DE RUTAS
if(WIN32)
    message(STATUS "Configurando para Windows")
    set(OpenCV_DIR "C:/dev/opencv/build_cpu")
else()
    message(STATUS "Configurando para Linux")
    set(OpenCV_DIR "/home/felipep/Documentos/universidad/universidad 7mo/vision por computador/opencv-dev/install/lib/cmake/opencv4")
endif()

# 2. BUSCAR DEPENDENCIAS
find_package(OpenCV REQUIRED)

# 3. DIRECTORIOS E INCLUDES
# Incluir src para que los #include funcionen correctamente
include_directories(${CMAKE_SOURCE_DIR}/src)
include_directories(${OpenCV_INCLUDE_DIRS})

# Macro para acceder a carpetas 'data' y 'models' desde el código
add_definitions(-DPROJECT_SOURCE_DIR="${CMAKE_SOURCE_DIR}")

# 4. DEFINICIÓN DE ARCHIVOS FUENTE
set(SOURCES
    src/main.cpp
    # Agregar aquí los archivos .cpp del proyecto
)

# 5. CREACIÓN DEL EJECUTABLE
add_executable(DeteccionPeatones ${SOURCES})

# 6. VINCULACIÓN DE LIBRERÍAS
target_link_libraries(DeteccionPeatones PRIVATE ${OpenCV_LIBS})

# 7. MENSAJES DE INFORMACIÓN
message(STATUS "PROYECTO: Detección de Peatones")
message(STATUS "Compilando ejecutable: DeteccionPeatones")
message(STATUS "OpenCV: ${OpenCV_VERSION}")
