cmake_minimum_required(VERSION 3.10)
if(POLICY CMP0167)
    cmake_policy(SET CMP0167 NEW)
endif()

project(WebSocketFaceRecognition)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Explicitly exclude conda paths to avoid ABI conflicts
list(REMOVE_ITEM CMAKE_SYSTEM_PREFIX_PATH "/home/naiscorp/miniconda3")
list(REMOVE_ITEM CMAKE_SYSTEM_LIBRARY_PATH "/home/naiscorp/miniconda3/lib")
set(CMAKE_IGNORE_PATH "/home/naiscorp/miniconda3;/home/naiscorp/miniconda3/lib;/home/naiscorp/miniconda3/include")

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src/dnn)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src/dnn/mtcnn)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src/postgres)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src/websocket)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src/recognizer)
include_directories(/usr/include/nlohmann)

# Find packages
find_package(OpenCV REQUIRED)
find_package(Boost REQUIRED COMPONENTS system thread)
find_package(PostgreSQL REQUIRED)
find_library(PQXX_LIB pqxx REQUIRED)

# Include directories
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${Boost_INCLUDE_DIRS})
include_directories(${PostgreSQL_INCLUDE_DIRS})

# ONNX Runtime paths
set(ONNXRUNTIME_ROOT "/usr/local/onnxruntime")
set(ONNXRUNTIME_INCLUDE_DIR "${ONNXRUNTIME_ROOT}/include")
set(ONNXRUNTIME_LIB_DIR "${ONNXRUNTIME_ROOT}/lib")

include_directories(${ONNXRUNTIME_INCLUDE_DIR})

# Find CUDA
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

# Filter out conda libraries from OpenCV
if(OpenCV_LIBS)
    set(FILTERED_OpenCV_LIBS "")
    foreach(lib ${OpenCV_LIBS})
        string(FIND "${lib}" "miniconda3" pos)
        if(pos EQUAL -1)
            list(APPEND FILTERED_OpenCV_LIBS ${lib})
        endif()
    endforeach()
    set(OpenCV_LIBS ${FILTERED_OpenCV_LIBS})
endif()

# Add ONNX Runtime library directory FIRST (before system paths)
link_directories(BEFORE ${ONNXRUNTIME_LIB_DIR})

# Explicitly exclude conda library paths from linker search
link_directories(BEFORE /usr/lib /usr/lib/x86_64-linux-gnu)

# DNN module sources (reuse from parent)
set(DNN_SOURCES
    src/dnn/mtcnn/detector.cpp
    src/dnn/mtcnn/onet.cpp 
    src/dnn/mtcnn/pnet.cpp
    src/dnn/mtcnn/rnet.cpp
    src/dnn/mtcnn/onnx_module.cpp
    src/dnn/draw.hpp
)

set(RECOGNIZER_SOURCES
    src/websocket/config.cpp
    src/websocket/face_quality.cpp
    src/recognizer/face_recognizer.cpp
    src/postgres/postgres.cpp
)

# Build executable
add_executable(websocket_server ${DNN_SOURCES} ${RECOGNIZER_SOURCES} websocket_server.cpp)

# Link libraries
target_link_libraries(websocket_server 
    ${OpenCV_LIBS} 
    ${Boost_LIBRARIES}
    onnxruntime 
    pqxx 
    pq
    pthread
)

# Set linker flags and RPATH
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--as-needed -Wl,-rpath,${ONNXRUNTIME_LIB_DIR}")
set_target_properties(websocket_server PROPERTIES
    INSTALL_RPATH "${ONNXRUNTIME_LIB_DIR}"
    BUILD_WITH_INSTALL_RPATH TRUE
)

add_executable(register ${RECOGNIZER_SOURCES} ${DNN_SOURCES} register.cpp)

target_link_libraries(register
    ${OpenCV_LIBS}
    onnxruntime
    pqxx
    pq
)

set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--as-needed -Wl,-rpath,${ONNXRUNTIME_LIB_DIR}")
set_target_properties(register PROPERTIES
    INSTALL_RPATH "${ONNXRUNTIME_LIB_DIR}"
    BUILD_WITH_INSTALL_RPATH TRUE
)

add_executable(recognize ${RECOGNIZER_SOURCES} ${DNN_SOURCES} recognize.cpp)

target_link_libraries(recognize
    ${OpenCV_LIBS}
    onnxruntime
    pqxx
    pq
)

set_target_properties(recognize PROPERTIES
    INSTALL_RPATH "${ONNXRUNTIME_LIB_DIR}"
    BUILD_WITH_INSTALL_RPATH TRUE
)