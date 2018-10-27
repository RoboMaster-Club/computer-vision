cmake_minimum_required(VERSION 3.12)
project(cascade_classifier_test)

set(CMAKE_CXX_STANDARD 11)
find_package(OpenCV REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")

add_executable(cascade_classifier_test main.cpp)
target_link_libraries(cascade_classifier_test ${OpenCV_LIBS})