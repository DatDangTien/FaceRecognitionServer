# Multi-stage build for Face Recognition API Server
FROM ubuntu:22.04 AS builder

# Avoid interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    pkg-config \
    libopencv-dev \
    libboost-system-dev \
    libboost-thread-dev \
    libpq-dev \
    libpqxx-dev \
    wget \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install ONNX Runtime
# Download and install ONNX Runtime for Linux x64
WORKDIR /tmp
RUN wget https://github.com/microsoft/onnxruntime/releases/download/v1.16.3/onnxruntime-linux-x64-1.16.3.tgz \
    && tar -xzf onnxruntime-linux-x64-1.16.3.tgz \
    && mkdir -p /usr/local/onnxruntime \
    && cp -r onnxruntime-linux-x64-1.16.3/* /usr/local/onnxruntime/ \
    && rm -rf onnxruntime-linux-x64-1.16.3.tgz onnxruntime-linux-x64-1.16.3

# Install nlohmann/json headers
RUN apt-get update && apt-get install -y \
    nlohmann-json3-dev \
    && rm -rf /var/lib/apt/lists/*

# Build the API server
WORKDIR /build
COPY CMakeLists.txt ./
COPY api ./api
COPY src ./src
COPY models ./models
COPY third_party/oatpp ./third_party/oatpp
# Copy other source files to satisfy CMake (even though we won't build them)
COPY websocket_server.cpp register.cpp recognize.cpp live_recognize.cpp ./

# Build Oat++ library (must be in the expected location for CMakeLists.txt)
# Clean any existing build artifacts and create fresh build directory
# Following Oat++ standard build method from README
WORKDIR /build/third_party/oatpp
RUN rm -rf build && mkdir -p build
WORKDIR /build/third_party/oatpp/build
RUN cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_SHARED_LIBS=OFF \
    -DOATPP_BUILD_TESTS=OFF \
    && make oatpp -j$(nproc)

# Create build directory and configure
WORKDIR /build/build
RUN cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    && cmake --build . --target api_server -j$(nproc)

# Runtime stage
FROM ubuntu:22.04

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install runtime dependencies
# Note: libopencv-dev includes both dev headers and runtime libraries
# For runtime-only, we can use libopencv-dev or install specific versioned packages
RUN apt-get update && apt-get install -y \
    libopencv-dev \
    libboost-system1.74.0 \
    libboost-thread1.74.0 \
    libpq5 \
    libpqxx-6.4 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy ONNX Runtime from builder
COPY --from=builder /usr/local/onnxruntime /usr/local/onnxruntime

# Copy built executable
COPY --from=builder /build/build/api_server /usr/local/bin/api_server

# Copy models (recursive by default)
COPY --from=builder /build/models /app/models
# Copy src directory structure recursively (includes dnn/models and all subdirectories)
COPY --from=builder /build/src /app/src

# Copy config file (user can override with volume mount in docker-compose.yml)
COPY config.ini /app/config.ini

# Create directory structure for output data
RUN mkdir -p /app/data/save_output

# Set working directory
WORKDIR /app

# Set environment variables
ENV LD_LIBRARY_PATH=/usr/local/onnxruntime/lib
ENV CONFIG_PATH=/app/config.ini
ENV API_HOST=0.0.0.0
ENV API_PORT=7997

# Expose port
EXPOSE 7997

# Run the server
CMD ["/usr/local/bin/api_server"]

