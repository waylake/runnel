cmake_minimum_required(VERSION 3.20)

project(runnel_native LANGUAGES CXX OBJCXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(runnel_metal_probe STATIC
  src/metal_probe.mm
  src/quantized_qmv_probe.mm
  src/quantized_qmm_probe.mm
  src/fused_moe_probe.mm
)

target_compile_features(runnel_metal_probe PUBLIC cxx_std_20)
target_include_directories(runnel_metal_probe PUBLIC
  "${CMAKE_CURRENT_SOURCE_DIR}/include"
)
target_compile_options(runnel_metal_probe PRIVATE
  -O3
  -Wall
  -Wextra
  -Wpedantic
  -fobjc-arc
)
target_compile_definitions(runnel_metal_probe PRIVATE
  RUNNEL_DEFAULT_METAL_SHADER="${CMAKE_CURRENT_SOURCE_DIR}/shaders/persistent_stream.metal"
  RUNNEL_DEFAULT_QUANTIZED_QMV_SHADER="${CMAKE_CURRENT_SOURCE_DIR}/shaders/quantized_qmv.metal"
  RUNNEL_DEFAULT_QUANTIZED_QMM_SHADER="${CMAKE_CURRENT_SOURCE_DIR}/shaders/quantized_qmm.metal"
  RUNNEL_DEFAULT_FUSED_MOE_SHADER="${CMAKE_CURRENT_SOURCE_DIR}/shaders/fused_moe.metal"
)
target_link_libraries(runnel_metal_probe PUBLIC
  "-framework Foundation"
  "-framework IOKit"
  "-framework Metal"
)

add_executable(runnel-metal-stream bench/metal_stream.cpp)
target_compile_features(runnel-metal-stream PRIVATE cxx_std_20)
target_compile_options(runnel-metal-stream PRIVATE
  -O3
  -Wall
  -Wextra
  -Wpedantic
)
target_link_libraries(runnel-metal-stream PRIVATE runnel_metal_probe)

add_executable(runnel-metal-qmv bench/quantized_qmv.cpp)
target_compile_features(runnel-metal-qmv PRIVATE cxx_std_20)
target_compile_options(runnel-metal-qmv PRIVATE
  -O3
  -Wall
  -Wextra
  -Wpedantic
)
target_link_libraries(runnel-metal-qmv PRIVATE runnel_metal_probe)

add_executable(runnel-metal-qmm bench/quantized_qmm.cpp)
target_compile_features(runnel-metal-qmm PRIVATE cxx_std_20)
target_compile_options(runnel-metal-qmm PRIVATE
  -O3
  -Wall
  -Wextra
  -Wpedantic
)
target_link_libraries(runnel-metal-qmm PRIVATE runnel_metal_probe)

add_executable(runnel-metal-fused-moe bench/fused_moe.cpp)
target_compile_features(runnel-metal-fused-moe PRIVATE cxx_std_20)
target_compile_options(runnel-metal-fused-moe PRIVATE
  -O3
  -Wall
  -Wextra
  -Wpedantic
)
target_link_libraries(runnel-metal-fused-moe PRIVATE runnel_metal_probe)

include(CTest)
if(BUILD_TESTING)
  add_test(
    NAME metal-stream-smoke
    COMMAND runnel-metal-stream --bytes-gib 0.0625 --warmups 0 --trials 1 --quick
  )
  add_test(
    NAME metal-qmv-smoke
    COMMAND runnel-metal-qmv --bits q4 --output-size 4096 --reduction-size 2048 --groups 32 --threads 256 --warmups 0 --trials 1
  )
  add_test(
    NAME metal-qmv-expert-gather-smoke
    COMMAND runnel-metal-qmv --bits q4 --output-size 512 --reduction-size 2048 --groups 32 --threads 128 --weight-bank-output-size 8192 --rows-per-expert 128 --experts 0,7,19,31 --warmups 0 --trials 1
  )
  add_test(
    NAME metal-qmm-4d-smoke
    COMMAND runnel-metal-qmm --bits q4 --tokens 8 --outputs 32 --reduction-size 2048 --warmups 0 --trials 1
  )
  add_test(
    NAME metal-fused-moe-smoke
    COMMAND runnel-metal-fused-moe --bits q4 --total-experts 16 --experts 0,1,2,3,4,5,6,7 --groups 32 --threads 256 --warmups 0 --trials 1
  )
  add_test(
    NAME metal-two-stage-moe-smoke
    COMMAND runnel-metal-fused-moe --bits q4 --total-experts 16 --experts 0,1,2,3,4,5,6,7 --groups 32 --threads 256 --warmups 0 --trials 1 --two-stage
  )
endif()
