cmake_minimum_required(VERSION 3.17)
project(FinalVision CXX)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)
find_package(CURL REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS} ${CURL_INCLUDE_DIRS})

add_executable(app_vigilante src/main.cpp)

target_link_libraries(app_vigilante PRIVATE ${OpenCV_LIBS} CURL::libcurl)