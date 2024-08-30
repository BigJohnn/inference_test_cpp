cmake_minimum_required(VERSION 3.10)
project(inference_test)

find_package(OpenCV REQUIRED)

add_definitions(-DROOT_DIR="${CMAKE_SOURCE_DIR}/")
set(Torch_DIR "~/Codes/libtorch/share/cmake/Torch")
find_package(Torch REQUIRED)

set(Torch_INCLUDE_DIRS "~/Codes/libtorch/include")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# 添加需要包含的头文件目录
include_directories(${OpenCV_INCLUDE_DIRS} ${Torch_INCLUDE_DIRS})

# 添加源文件
add_executable(inference_test main.cpp)
set_property(TARGET inference_test PROPERTY CXX_STANDARD 17)

# 添加链接库
target_link_libraries(inference_test ${OpenCV_LIBS} ${TORCH_LIBRARIES})