cmake_minimum_required(VERSION 3.16)
project(FinalVision CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Configuración de rutas según el sistema
if(WIN32)
    message(STATUS "Configurando para Windows")
    set(OpenCV_DIR "C:/dev/opencv/build_cpu")
else()
    message(STATUS "Configurando para Linux")
    # OpenCV debería encontrarse automáticamente en Linux
    # Si no, descomenta y ajusta la siguiente línea:
    # set(OpenCV_DIR "/usr/local/lib/cmake/opencv4")
endif()

# Buscar dependencias
find_package(OpenCV REQUIRED)
find_package(CURL REQUIRED)

# Incluir directorios
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${CURL_INCLUDE_DIRS})

# Definir macro para acceder a la carpeta del proyecto
add_definitions(-DPROJECT_SOURCE_DIR="${CMAKE_SOURCE_DIR}")

# Crear ejecutable LBP simple (original - solo personas paradas)
add_executable(app_vigilante src/main.cpp)
target_link_libraries(app_vigilante PRIVATE ${OpenCV_LIBS} CURL::libcurl)

# Crear ejecutable HOG (alternativo - solo personas verticales)
add_executable(app_hog src/main_hog.cpp)
target_link_libraries(app_hog PRIVATE ${OpenCV_LIBS} CURL::libcurl)

# Crear ejecutable DUAL (LBP doble - experimental)
add_executable(app_dual src/main_dual.cpp)
target_link_libraries(app_dual PRIVATE ${OpenCV_LIBS} CURL::libcurl)

# Crear ejecutable HYBRID (RECOMENDADO - HOG+LBP)
add_executable(app_hybrid src/main_hybrid.cpp)
target_link_libraries(app_hybrid PRIVATE ${OpenCV_LIBS} CURL::libcurl)

# Crear ejecutable HYBRID GPU (GPU-Accelerated con CUDA)
add_executable(app_hybrid_gpu src/main_hybrid_gpu.cpp)
target_link_libraries(app_hybrid_gpu PRIVATE ${OpenCV_LIBS} CURL::libcurl)

# Copiar archivos necesarios al build (los modelos .xml)
file(COPY ${CMAKE_SOURCE_DIR}/../training/models/standing/cascade_standing.xml 
     DESTINATION ${CMAKE_BINARY_DIR})
file(COPY ${CMAKE_SOURCE_DIR}/../training/models/complex/cascade_crouching.xml 
     DESTINATION ${CMAKE_BINARY_DIR})

# Mensajes informativos
message(STATUS "PROYECTO: Detección de Peatones")
message(STATUS "Ejecutables:")
message(STATUS "  - app_hybrid_gpu (HOG GPU + LBP - ULTRA RAPIDO)")
message(STATUS "  - app_hybrid (HOG+LBP - RECOMENDADO)")
message(STATUS "  - app_dual (LBP Dual - experimental)")
message(STATUS "  - app_hog (HOG + SVM)")
message(STATUS "  - app_vigilante (LBP simple)")
message(STATUS "OpenCV: ${OpenCV_VERSION}")
message(STATUS "Build directory: ${CMAKE_BINARY_DIR}")
message(STATUS "")
message(STATUS "Los modelos .xml se copiarán automáticamente a build/")
message(STATUS "")
message(STATUS "Para ejecutar:")
message(STATUS "  HYBRID GPU (CUDA):     ./app_hybrid_gpu  <-- MÁS RÁPIDO")
message(STATUS "  HYBRID (HOG+LBP):      ./app_hybrid  <-- RECOMENDADO")
message(STATUS "  DUAL (todas posturas): ./app_dual")
message(STATUS "  HOG (solo parados):    ./app_hog")
message(STATUS "  LBP simple:            ./app_vigilante")