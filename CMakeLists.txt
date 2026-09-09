cmake_minimum_required(VERSION 3.20)
project(TinyLogicVision VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(tinyvision_core
    src/model.cpp
    src/dataset.cpp
    src/observation.cpp
    src/spatial_experiment.cpp
)
target_include_directories(tinyvision_core PUBLIC include)
target_compile_options(tinyvision_core PRIVATE -Wall -Wextra -Wpedantic)

add_executable(tinyvision src/main.cpp)
target_link_libraries(tinyvision PRIVATE tinyvision_core)

add_executable(tinyvision_generalization src/generalization.cpp)
target_link_libraries(tinyvision_generalization PRIVATE tinyvision_core)

add_executable(tinyvision_spatial_coverage src/spatial_coverage.cpp)
target_link_libraries(tinyvision_spatial_coverage PRIVATE tinyvision_core)

enable_testing()

add_executable(gradient_test tests/gradient_test.cpp)
target_link_libraries(gradient_test PRIVATE tinyvision_core)
add_test(NAME gradient_test COMMAND gradient_test)

add_executable(training_test tests/training_test.cpp)
target_link_libraries(training_test PRIVATE tinyvision_core)
add_test(NAME training_test COMMAND training_test)

add_executable(generalization_observatory_test tests/generalization_observatory_test.cpp)
target_link_libraries(generalization_observatory_test PRIVATE tinyvision_core)
add_test(NAME generalization_observatory_test COMMAND generalization_observatory_test)

add_executable(spatial_coverage_experiment_test tests/spatial_coverage_experiment_test.cpp)
target_link_libraries(spatial_coverage_experiment_test PRIVATE tinyvision_core)
add_test(NAME spatial_coverage_experiment_test COMMAND spatial_coverage_experiment_test)
