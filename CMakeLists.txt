cmake_minimum_required(VERSION 3.17)
project(coursework)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_PREFIX_PATH "/Users/ulanovgeorgiy/dev/open_cv/opencv/build")
set(OpenCV_DIR "/Users/ulanovgeorgiy/dev/open_cv/opencv")

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

message("MESSAGE")
message(${OpenCV_LIBS})

add_executable(coursework main.cpp utils.cpp)
target_link_libraries(coursework ${OpenCV_LIBS})
