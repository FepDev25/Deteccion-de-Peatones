cmake_minimum_required(VERSION 3.16)
project(PedestrianDetectionQt)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

# Evitar conflicto con librerías de Anaconda
set(CMAKE_SKIP_BUILD_RPATH FALSE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
set(CMAKE_INSTALL_RPATH "/usr/lib")

# Priorizar librerías del sistema sobre las de Anaconda
list(REMOVE_ITEM CMAKE_SYSTEM_PREFIX_PATH "$ENV{CONDA_PREFIX}")
list(REMOVE_ITEM CMAKE_SYSTEM_PREFIX_PATH "$ENV{HOME}/anaconda3")
list(REMOVE_ITEM CMAKE_PREFIX_PATH "$ENV{CONDA_PREFIX}")
list(REMOVE_ITEM CMAKE_PREFIX_PATH "$ENV{HOME}/anaconda3")

# Buscar Qt6 (o Qt5 como fallback)
find_package(Qt6 QUIET COMPONENTS Core Widgets)
if(NOT Qt6_FOUND)
    find_package(Qt5 REQUIRED COMPONENTS Core Widgets)
    set(QT_VERSION ${Qt5_VERSION})
    set(QT_LIBS Qt5::Core Qt5::Widgets)
else()
    set(QT_VERSION ${Qt6_VERSION})
    set(QT_LIBS Qt6::Core Qt6::Widgets)
endif()

# Buscar OpenCV
find_package(OpenCV REQUIRED)

# Buscar CURL
find_package(CURL REQUIRED)

# Incluir directorios
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${CURL_INCLUDE_DIRS})
include_directories(${Qt${QT_VERSION_MAJOR}Widgets_INCLUDE_DIRS})

# Crear ejecutable Qt GUI
add_executable(app_qt_gui
    src/main_qt.cpp
    src/mainwindow.cpp
    src/mainwindow.h
)

# Configurar RPATH para evitar conflictos con Anaconda
set_target_properties(app_qt_gui PROPERTIES
    INSTALL_RPATH "/usr/lib"
    BUILD_WITH_INSTALL_RPATH FALSE
    SKIP_BUILD_RPATH FALSE
)

# Linkear bibliotecas
target_link_libraries(app_qt_gui
    ${OpenCV_LIBS}
    CURL::libcurl
    ${QT_LIBS}
)

# Copiar modelos XML
file(COPY ${CMAKE_SOURCE_DIR}/../training/models/standing/cascade_standing.xml 
     DESTINATION ${CMAKE_BINARY_DIR})
file(COPY ${CMAKE_SOURCE_DIR}/../training/models/complex/cascade_crouching.xml 
     DESTINATION ${CMAKE_BINARY_DIR})

message(STATUS "===========================================")
message(STATUS "  PROYECTO QT: Detección de Peatones")
message(STATUS "===========================================")
message(STATUS "Qt Version: ${QT_VERSION}")
message(STATUS "OpenCV: ${OpenCV_VERSION}")
message(STATUS "Build directory: ${CMAKE_BINARY_DIR}")
message(STATUS "")
message(STATUS "Para ejecutar:")
message(STATUS "  ./app_qt_gui")
message(STATUS "===========================================")
