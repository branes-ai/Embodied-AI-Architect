cmake_minimum_required(VERSION 3.14)
project(drone_perception VERSION 1.0)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)

add_executable(drone_app
    src/main.cpp
    src/perception.cpp
    src/control.cpp
)

target_link_libraries(drone_app
    PUBLIC
    opencv_core
    opencv_imgproc
    Eigen3::Eigen
)
