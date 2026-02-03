cmake_minimum_required(VERSION 3.16)
project(PedestrianDetectionQt)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

# Buscar Qt5
find_package(Qt5 REQUIRED COMPONENTS Core Widgets)

# Buscar OpenCV
find_package(OpenCV REQUIRED)

# Buscar CURL
find_package(CURL REQUIRED)

# Incluir directorios
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${CURL_INCLUDE_DIRS})
include_directories(${Qt5Widgets_INCLUDE_DIRS})

# Crear ejecutable Qt GUI
add_executable(app_qt_gui
    src/main_qt.cpp
    src/mainwindow.cpp
    src/mainwindow.h
)

# Linkear bibliotecas
target_link_libraries(app_qt_gui
    ${OpenCV_LIBS}
    CURL::libcurl
    Qt5::Core
    Qt5::Widgets
)

# Copiar modelos XML
file(COPY ${CMAKE_SOURCE_DIR}/../training/models/standing/cascade_standing.xml 
     DESTINATION ${CMAKE_BINARY_DIR})
file(COPY ${CMAKE_SOURCE_DIR}/../training/models/complex/cascade_crouching.xml 
     DESTINATION ${CMAKE_BINARY_DIR})

message(STATUS "===========================================")
message(STATUS "  PROYECTO QT: Detección de Peatones")
message(STATUS "===========================================")
message(STATUS "Qt Version: ${Qt5_VERSION}")
message(STATUS "OpenCV: ${OpenCV_VERSION}")
message(STATUS "Build directory: ${CMAKE_BINARY_DIR}")
message(STATUS "")
message(STATUS "Para ejecutar:")
message(STATUS "  ./app_qt_gui")
message(STATUS "===========================================")
