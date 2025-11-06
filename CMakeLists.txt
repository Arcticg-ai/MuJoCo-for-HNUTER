cmake_minimum_required(VERSION 3.10)
cmake_policy(SET CMP0072 NEW)

project(mujoco_client)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

find_package(glfw3 REQUIRED)
find_package(mujoco REQUIRED)
find_package(Eigen3 REQUIRED)

include_directories(
        include
        ${EIGEN3_INCLUDE_DIR})

add_library(mujoco_client src/mujoco_client.cpp)
target_link_libraries(mujoco_client mujoco glfw OpenGL::GL)

add_executable(collision_test src/collision_checker_test.cpp)
add_dependencies(collision_test mujoco_client)
target_link_libraries(collision_test mujoco_client)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
