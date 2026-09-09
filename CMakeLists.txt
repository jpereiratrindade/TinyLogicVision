cmake_minimum_required(VERSION 3.20)
project(TinyLogicVision VERSION 0.1.0 LANGUAGES CXX)

find_package(PNG REQUIRED)
find_package(JPEG REQUIRED)

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

add_library(tinyvision_application
    src/image.cpp
    src/application.cpp
    src/dense.cpp
)
target_include_directories(tinyvision_application PUBLIC include)
target_link_libraries(tinyvision_application PUBLIC tinyvision_core PNG::PNG JPEG::JPEG)
target_compile_options(tinyvision_application PRIVATE -Wall -Wextra -Wpedantic)

add_executable(tinyvision src/main.cpp)
target_link_libraries(tinyvision PRIVATE tinyvision_core)

add_executable(tinyvision_generalization src/generalization.cpp)
target_link_libraries(tinyvision_generalization PRIVATE tinyvision_core)

add_executable(tinyvision_spatial_coverage src/spatial_coverage.cpp)
target_link_libraries(tinyvision_spatial_coverage PRIVATE tinyvision_core)

add_executable(tinyvision_train src/application_train.cpp)
target_link_libraries(tinyvision_train PRIVATE tinyvision_application)

add_executable(tinyvision_classify src/application_classify.cpp)
target_link_libraries(tinyvision_classify PRIVATE tinyvision_application)

add_executable(tinyvision_evaluate src/application_evaluate.cpp)
target_link_libraries(tinyvision_evaluate PRIVATE tinyvision_application)

add_executable(tinyvision_map src/application_map.cpp)
target_link_libraries(tinyvision_map PRIVATE tinyvision_application)

enable_testing()

add_executable(gradient_test tests/gradient_test.cpp)
target_link_libraries(gradient_test PRIVATE tinyvision_core)
add_test(NAME gradient_test COMMAND gradient_test)

add_executable(core_perf_test tests/core_perf_test.cpp)
target_link_libraries(core_perf_test PRIVATE tinyvision_core)
add_test(NAME core_perf_test COMMAND core_perf_test)

add_executable(schema_model_test tests/schema_model_test.cpp)
target_link_libraries(schema_model_test PRIVATE tinyvision_application)
add_test(NAME schema_model_test COMMAND schema_model_test)

add_executable(training_test tests/training_test.cpp)
target_link_libraries(training_test PRIVATE tinyvision_core)
add_test(NAME training_test COMMAND training_test)

add_executable(generalization_observatory_test tests/generalization_observatory_test.cpp)
target_link_libraries(generalization_observatory_test PRIVATE tinyvision_core)
add_test(NAME generalization_observatory_test COMMAND generalization_observatory_test)

add_executable(spatial_coverage_experiment_test tests/spatial_coverage_experiment_test.cpp)
target_link_libraries(spatial_coverage_experiment_test PRIVATE tinyvision_core)
add_test(NAME spatial_coverage_experiment_test COMMAND spatial_coverage_experiment_test)

add_executable(application_pipeline_test tests/application_pipeline_test.cpp)
target_link_libraries(application_pipeline_test PRIVATE tinyvision_application)
add_test(NAME application_pipeline_test COMMAND application_pipeline_test)

add_executable(dense_test tests/dense_test.cpp)
target_link_libraries(dense_test PRIVATE tinyvision_application)
add_test(NAME dense_test COMMAND dense_test)

add_test(NAME cli_test COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/tests/cli_test.sh")
add_test(NAME web_test COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/tests/web_test.py")
