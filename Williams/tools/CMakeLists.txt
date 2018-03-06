cmake_minimum_required(VERSION 3.5)
project(tools)

find_package(OpenCV REQUIRED)

set(CMAKE_CXX_STANDARD 11)

add_executable(tools main.cpp)
target_link_libraries(tools ${OpenCV_LIBS})
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")