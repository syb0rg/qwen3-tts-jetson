cmake_minimum_required(VERSION 3.14)
project(qwen3-tts-ggml VERSION 0.1.0 LANGUAGES CXX)

# C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Compiler flags
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=native")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -O0")
endif()

# CoreML code predictor bridge (macOS only)
option(QWEN3_TTS_COREML "Enable CoreML code predictor bridge on macOS" ON)
if(APPLE AND QWEN3_TTS_COREML)
    enable_language(OBJCXX)
endif()

# TensorRT vocoder acceleration
option(QWEN3_TTS_TENSORRT "Enable TensorRT vocoder acceleration" OFF)
if(QWEN3_TTS_TENSORRT)
    find_package(CUDAToolkit REQUIRED)
    find_library(NVINFER_LIB nvinfer HINTS /usr/lib/aarch64-linux-gnu)
    find_path(NVINFER_INCLUDE NvInfer.h HINTS /usr/include/aarch64-linux-gnu /usr/include)
    if(NOT NVINFER_LIB)
        message(FATAL_ERROR "libnvinfer not found. Install TensorRT dev packages or set -DQWEN3_TTS_TENSORRT=OFF.")
    endif()
    if(NOT NVINFER_INCLUDE)
        message(FATAL_ERROR "NvInfer.h not found. Install TensorRT dev packages or set -DQWEN3_TTS_TENSORRT=OFF.")
    endif()
    message(STATUS "TensorRT lib: ${NVINFER_LIB}")
    message(STATUS "TensorRT include: ${NVINFER_INCLUDE}")
endif()

# Find threads
find_package(Threads REQUIRED)

# GGML library paths
set(GGML_DIR "${CMAKE_CURRENT_SOURCE_DIR}/ggml")
set(GGML_BUILD_DIR "${GGML_DIR}/build")

# Text tokenizer library
add_library(text_tokenizer STATIC
    src/text_tokenizer.cpp
    src/gguf_loader.cpp
)
target_include_directories(text_tokenizer PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${GGML_DIR}/include
)
target_link_directories(text_tokenizer PUBLIC
    ${GGML_BUILD_DIR}/src
)
target_link_libraries(text_tokenizer PUBLIC
    ggml
    ggml-base
    ggml-cpu
    Threads::Threads
)

# TTS Transformer library (GGML-based + optional CoreML bridge)
set(TTS_TRANSFORMER_SOURCES
    src/gguf_loader.cpp
    src/tts_transformer.cpp
)
if(APPLE AND QWEN3_TTS_COREML)
    list(APPEND TTS_TRANSFORMER_SOURCES src/coreml_code_predictor.mm)
    set_source_files_properties(src/coreml_code_predictor.mm PROPERTIES COMPILE_FLAGS "-fobjc-arc")
else()
    list(APPEND TTS_TRANSFORMER_SOURCES src/coreml_code_predictor_stub.cpp)
endif()

add_library(tts_transformer STATIC
    ${TTS_TRANSFORMER_SOURCES}
)
target_include_directories(tts_transformer PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${GGML_DIR}/include
)
target_link_directories(tts_transformer PUBLIC
    ${GGML_BUILD_DIR}/src
)
target_link_libraries(tts_transformer PUBLIC
    ggml
    ggml-base
    ggml-cpu
    Threads::Threads
)
if(APPLE AND QWEN3_TTS_COREML)
    target_link_libraries(tts_transformer PUBLIC
        "-framework Foundation"
        "-framework CoreML"
    )
endif()
if(QWEN3_TTS_TENSORRT)
    enable_language(CUDA)
    target_sources(tts_transformer PRIVATE src/trt_code_predictor.cpp src/trt_cuda_kernels.cu)
    target_compile_definitions(tts_transformer PUBLIC QWEN3_TTS_TENSORRT)
    target_include_directories(tts_transformer PUBLIC ${NVINFER_INCLUDE})
    target_link_libraries(tts_transformer PUBLIC ${NVINFER_LIB} CUDA::cudart CUDA::cublas)
endif()

# Audio tokenizer encoder library (GGML-based)
add_library(audio_tokenizer_encoder STATIC
    src/audio_tokenizer_encoder.cpp
    src/gguf_loader.cpp
)
target_include_directories(audio_tokenizer_encoder PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${GGML_DIR}/include
)
target_link_directories(audio_tokenizer_encoder PUBLIC
    ${GGML_BUILD_DIR}/src
)
target_link_libraries(audio_tokenizer_encoder PUBLIC
    ggml
    ggml-base
    ggml-cpu
    Threads::Threads
)

# Audio tokenizer decoder library (GGML-based)
add_library(audio_tokenizer_decoder STATIC
    src/audio_tokenizer_decoder.cpp
    src/gguf_loader.cpp
)
target_include_directories(audio_tokenizer_decoder PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${GGML_DIR}/include
)
target_link_directories(audio_tokenizer_decoder PUBLIC
    ${GGML_BUILD_DIR}/src
)
target_link_libraries(audio_tokenizer_decoder PUBLIC
    ggml
    ggml-base
    ggml-cpu
    Threads::Threads
)

# Qwen3 TTS library (full pipeline)
add_library(qwen3_tts STATIC
    src/qwen3_tts.cpp
)
target_include_directories(qwen3_tts PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${GGML_DIR}/include
)
target_link_libraries(qwen3_tts PUBLIC
    text_tokenizer
    tts_transformer
    audio_tokenizer_encoder
    audio_tokenizer_decoder
    Threads::Threads
)
if(QWEN3_TTS_TENSORRT)
    target_sources(qwen3_tts PRIVATE src/trt_vocoder.cpp)
    target_compile_definitions(qwen3_tts PUBLIC QWEN3_TTS_TENSORRT)
    target_include_directories(qwen3_tts PUBLIC ${NVINFER_INCLUDE})
    target_link_libraries(qwen3_tts PUBLIC ${NVINFER_LIB} CUDA::cudart)
endif()

# CLI executable
add_executable(qwen3-tts-cli
    src/main.cpp
)
target_link_libraries(qwen3-tts-cli PRIVATE
    qwen3_tts
)

# Test executable for tokenizer
add_executable(test_tokenizer
    tests/test_tokenizer.cpp
)
target_link_libraries(test_tokenizer PRIVATE
    text_tokenizer
    Threads::Threads
)

# Install targets
install(TARGETS text_tokenizer tts_transformer audio_tokenizer_encoder audio_tokenizer_decoder qwen3_tts qwen3-tts-cli
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)
install(FILES 
    src/gguf_loader.h 
    src/text_tokenizer.h 
    src/tts_transformer.h 
    src/coreml_code_predictor.h
    src/audio_tokenizer_encoder.h 
    src/audio_tokenizer_decoder.h 
    src/qwen3_tts.h
    DESTINATION include
)

# CTest support
enable_testing()
add_test(NAME tokenizer_test
    COMMAND test_tokenizer
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
