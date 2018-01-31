cmake_minimum_required(VERSION 3.5)
project(knn_recgonizer)

set(CMAKE_CXX_STANDARD 11)

set(OpenCV_DIR /usr/local/share/OpenCV)
find_package(OpenCV REQUIRED)

set(SOURCE_FILES main.cpp map_pgm.cpp map_pgm.h)

add_executable(knn_recgonizer ${SOURCE_FILES})

target_link_libraries(knn_recgonizer ${OpenCV_LIBS})
