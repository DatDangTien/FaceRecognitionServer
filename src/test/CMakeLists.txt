cmake_minimum_required(VERSION 3.15)
project(test_onnx)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Find LibTorch
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# Set ONNX Runtime paths
set(ONNXRUNTIME_ROOT "/usr/local/onnxruntime")
set(ONNXRUNTIME_INCLUDE_DIR "${ONNXRUNTIME_ROOT}/include")
set(ONNXRUNTIME_LIB_DIR "${ONNXRUNTIME_ROOT}/lib")

# Include ONNX Runtime headers
include_directories(${ONNXRUNTIME_INCLUDE_DIR})

# Find CUDA
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

# Add parent directory to include path
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/..)

# MTCNN library sources
set(MTCNN_SOURCES
    ../onnx/mtcnn.cpp
    ../onnx/mtcnn_utils.cpp
)

# Create executable for basic ONNX test
add_executable(test_onnx test_onnx.cpp)

# Create executable for MTCNN test
add_executable(test_mtcnn test_mtcnn.cpp ${MTCNN_SOURCES})

# Link ONNX Runtime libraries for test_onnx
target_link_libraries(test_onnx
    ${OpenCV_LIBS}
    ${TORCH_LIBRARIES}
    ${ONNXRUNTIME_LIB_DIR}/libonnxruntime.so
    ${ONNXRUNTIME_LIB_DIR}/libonnxruntime_providers_cuda.so
    ${ONNXRUNTIME_LIB_DIR}/libonnxruntime_providers_shared.so
    ${CUDA_LIBRARIES}
    ${CUDA_CUDART_LIBRARY}
)

# Link ONNX Runtime and LibTorch libraries for test_mtcnn
target_link_libraries(test_mtcnn
    ${OpenCV_LIBS}
    ${TORCH_LIBRARIES}
    ${ONNXRUNTIME_LIB_DIR}/libonnxruntime.so
    ${ONNXRUNTIME_LIB_DIR}/libonnxruntime_providers_cuda.so
    ${ONNXRUNTIME_LIB_DIR}/libonnxruntime_providers_shared.so
    ${CUDA_LIBRARIES}
    ${CUDA_CUDART_LIBRARY}
)

# Set RPATH to find shared libraries at runtime
set_target_properties(test_onnx PROPERTIES
    BUILD_RPATH "${ONNXRUNTIME_LIB_DIR}"
    INSTALL_RPATH "${ONNXRUNTIME_LIB_DIR}"
    INSTALL_RPATH_USE_LINK_PATH TRUE
)

set_target_properties(test_mtcnn PROPERTIES
    BUILD_RPATH "${ONNXRUNTIME_LIB_DIR}"
    INSTALL_RPATH "${ONNXRUNTIME_LIB_DIR}"
    INSTALL_RPATH_USE_LINK_PATH TRUE
)

