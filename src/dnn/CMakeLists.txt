cmake_minimum_required(VERSION 3.10)
if(POLICY CMP0167)
    cmake_policy(SET CMP0167 NEW)
endif()

project (MTCNN_APP)

set (CMAKE_CXX_STANDARD 14)
set (CMAKE_CXX_STANDARD_REQUIRED ON)

# Explicitly exclude conda paths to avoid ABI conflicts
list(REMOVE_ITEM CMAKE_SYSTEM_PREFIX_PATH "/home/naiscorp/miniconda3")
list(REMOVE_ITEM CMAKE_SYSTEM_LIBRARY_PATH "/home/naiscorp/miniconda3/lib")
set(CMAKE_IGNORE_PATH "/home/naiscorp/miniconda3;/home/naiscorp/miniconda3/lib;/home/naiscorp/miniconda3/include")

find_package( OpenCV REQUIRED )
find_package( Boost REQUIRED )
include_directories( ${Boost_INCLUDE_DIRS} )

# Find PostgreSQL and libpqxx
find_package(PostgreSQL REQUIRED)
find_library(PQXX_LIB pqxx REQUIRED)
include_directories(${PostgreSQL_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../postgres)

# Set ONNX Runtime paths
set(ONNXRUNTIME_ROOT "/usr/local/onnxruntime")
set(ONNXRUNTIME_INCLUDE_DIR "${ONNXRUNTIME_ROOT}/include")
set(ONNXRUNTIME_LIB_DIR "${ONNXRUNTIME_ROOT}/lib")

# Include ONNX Runtime headers
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

# Add linker flags to resolve undefined reference issues and exclude conda paths
# Set RPATH to ensure the correct ONNX Runtime library is used at runtime
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--as-needed -Wl,-rpath,${ONNXRUNTIME_LIB_DIR}")

# Explicitly exclude conda library paths from linker search
link_directories(BEFORE /usr/lib /usr/lib/x86_64-linux-gnu)

set(SOURCES
./mtcnn/detector.cpp
./mtcnn/face.h 
./mtcnn/onet.cpp 
./mtcnn/pnet.cpp ./mtcnn/rnet.cpp
./mtcnn/detector.h ./mtcnn/helpers.h ./mtcnn/onet.h ./mtcnn/pnet.h ./mtcnn/rnet.h
./draw.hpp
./mtcnn/onnx_module.cpp
../postgres/postgres.hpp
)

set(APP_SOURCES
infer_photo.cpp
infer_cam.cpp
)

# Build standard executables (without PostgreSQL)
foreach( sourcefile ${APP_SOURCES} )
    # Simple string replace, to cut off .cpp.
    string( REPLACE ".cpp" "" name ${sourcefile} )
    add_executable( ${name} ${sourcefile} ${SOURCES} )
    target_link_libraries (${name} LINK_PUBLIC ${OpenCV_LIBS} ${Boost_LIBRARIES} onnxruntime )
endforeach( sourcefile ${APP_SOURCES} )

# Build embed_face separately with PostgreSQL libraries
# Link order matters: pqxx before pq
add_executable( embed_face embed_face.cpp ${SOURCES} )
target_link_libraries (embed_face LINK_PUBLIC ${OpenCV_LIBS} ${Boost_LIBRARIES} onnxruntime pqxx pq )
