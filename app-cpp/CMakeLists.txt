cmake_minimum_required(VERSION 3.16)
project(DeteccionPeatones CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


set(RUTA_FELIPE "/home/felipep/Documentos/universidad/universidad 7mo/vision por computador/opencv-dev/install/lib/cmake/opencv4")

if(EXISTS "${RUTA_FELIPE}")
    message(STATUS ">> Detectado entorno de Felipe. Usando ruta personalizada.")
    set(OpenCV_DIR "${RUTA_FELIPE}")
else()
    message(STATUS ">> Entorno de Felipe no detectado. Buscando OpenCV en tu sistema...")
endif()

find_package(OpenCV REQUIRED)

include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${OpenCV_INCLUDE_DIRS})
add_definitions(-DPROJECT_SOURCE_DIR="${CMAKE_SOURCE_DIR}")

file(GLOB SOURCES "src/*.cpp")
add_executable(DeteccionPeatones ${SOURCES})

target_link_libraries(DeteccionPeatones PRIVATE ${OpenCV_LIBS})

message(STATUS "-------------------------------------------------")
message(STATUS " CONFIGURACIÓN CORRECTA (LINUX) ")
message(STATUS " Proyecto: DeteccionPeatones")
message(STATUS " OpenCV Versión: ${OpenCV_VERSION}")
message(STATUS " Ruta usada: ${OpenCV_DIR}")
message(STATUS "-------------------------------------------------")