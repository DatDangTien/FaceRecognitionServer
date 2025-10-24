# Face Recognition WebSocket Server

A high-performance C++ WebSocket server for real-time face recognition using MTCNN for face detection and ONNX Runtime for face embeddings. The server provides a WebSocket API for processing video frames and returning face recognition results with quality assessment.

## Features

- **Real-time Face Recognition**: Process video frames via WebSocket API
- **MTCNN Face Detection**: Multi-task CNN for accurate face detection and alignment
- **ONNX Runtime Integration**: Fast inference using ONNX Runtime for face embeddings
- **PostgreSQL Database**: Store and query face embeddings with similarity search
- **Face Quality Assessment**: Automatic quality filtering (blur, pose, lighting)
- **Multi-threaded Processing**: Asynchronous WebSocket handling with thread pool
- **Base64 Image Support**: Direct image processing from base64-encoded frames
- **Comprehensive Statistics**: Real-time processing metrics and performance monitoring

## Architecture

```
┌─────────────────┐    WebSocket    ┌──────────────────┐
│   Client App    │◄──────────────►│  WebSocket Server│
└─────────────────┘                 └──────────────────┘
                                             │
                                             ▼
                                    ┌──────────────────┐
                                    │   Face Pipeline  │
                                    │  ┌─────────────┐ │
                                    │  │   MTCNN     │ │
                                    │  │  Detection  │ │
                                    │  └─────────────┘ │
                                    │  ┌─────────────┐ │
                                    │  │   Quality   │ │
                                    │  │  Assessment │ │
                                    │  └─────────────┘ │
                                    │  ┌─────────────┐ │
                                    │  │   ONNX      │ │
                                    │  │  Embedding  │ │
                                    │  └─────────────┘ │
                                    └──────────────────┘
                                             │
                                             ▼
                                    ┌──────────────────┐
                                    │   PostgreSQL     │
                                    │   Database       │
                                    └──────────────────┘
```

## Dependencies

### System Requirements
- **Ubuntu 20.04+** (tested on Ubuntu 20.04/22.04)
- **CMake 3.10+**
- **C++14 compatible compiler** (GCC 7+ or Clang 5+)
- **CUDA 11.0+** (for GPU acceleration)

### Required Libraries
- **OpenCV 4.x** - Computer vision operations
- **Boost 1.65+** - WebSocket and networking (system, thread)
- **PostgreSQL 12+** - Database backend
- **libpqxx** - PostgreSQL C++ client library
- **ONNX Runtime 1.12+** - Neural network inference
- **nlohmann/json** - JSON parsing and generation

## Installation

### 1. Install System Dependencies

```bash
# Update package list
sudo apt update

# Install build tools and dependencies
sudo apt install -y \
    build-essential \
    cmake \
    git \
    pkg-config \
    libopencv-dev \
    libboost-all-dev \
    postgresql-server-dev-all \
    libpqxx-dev \
    nlohmann-json3-dev

# Install CUDA (optional, for GPU acceleration)
# Follow NVIDIA CUDA installation guide for your system
```

### 2. Install ONNX Runtime

```bash
# Download ONNX Runtime (CPU version)
wget https://github.com/microsoft/onnxruntime/releases/download/v1.15.1/onnxruntime-linux-x64-1.15.1.tgz

# Extract and install
tar -xzf onnxruntime-linux-x64-1.15.1.tgz
sudo mv onnxruntime-linux-x64-1.15.1 /usr/local/onnxruntime

# Add to library path
echo "/usr/local/onnxruntime/lib" | sudo tee /etc/ld.so.conf.d/onnxruntime.conf
sudo ldconfig
```

### 3. Clone and Build

```bash
# Clone the repository
git clone https://github.com/DatDangTien/VisionWake.git
cd VisionWake

# Create build directory
mkdir build && cd build

# Configure with CMake
cmake ..

# Build the project
make -j$(nproc)

# The executable will be created as 'websocket_server'
```

## Configuration

Edit `config.ini` to configure the server:

```ini

# Recognition Settings
recognition_threshold=0.3
detection_interval=5

# Quality Thresholds
blur_threshold=100.0
min_face_size=60.0
dark_ratio_threshold=0.4
bright_ratio_threshold=0.3
pose_threshold=1000.0
quality_threshold=0.5

# Server Settings
server_host=0.0.0.0
server_port=8764

# Model Paths
models_path=./src/dnn/models
inception_model_path=./models/inception.onnx
```

## Database Setup

### 1. Install PostgreSQL

```bash
sudo apt install postgresql postgresql-contrib
sudo systemctl start postgresql
sudo systemctl enable postgresql
```

### 2. Create Database and User

```bash
# Switch to postgres user
sudo -u postgres psql
```

## Usage
### 1. Register a person

```bash
# From the project root directory
./build/register "John Doe" /path/to/image.jpg
# Test the recognition
./build/recognize /path/to/image1.jpg
```
### 2. Start the Server
```bash
# From the project root directory
./build/websocket_server

# Or with custom config file
./build/websocket_server /path/to/config.ini
```

### 3. WebSocket API

Connect to `ws://localhost:8764` and send JSON messages:

#### Frame Processing
```json
{
    "type": "frame",
    "frame_id": 123,
    "frame": "base64_encoded_image_data",
    "tracker_id": 1,
    "bbox": {
        "xmin": 100,
        "ymin": 100
    }
}
```

#### Get Statistics
```json
{
    "type": "get_stats"
}
```

#### Response Format
```json
{
    "type": "result",
    "frame_id": 123,
    "recognition_results": [
        {
            "name": "John Doe",
            "confidence": 95.5,
            "status": "recognized",
            "bbox": {
                "xmin": 100.0,
                "ymin": 100.0,
                "xmax": 200.0,
                "ymax": 200.0
            },
            "tracker_id": 1
        }
    ],
    "processing_time": 0.045,
    "tracker_id": 1,
    "stats": {
        "total_frames_processed": 1500,
        "total_inferences_run": 1200,
        "successful_recognitions": 800,
        "quality_rejections": 300,
        "similarity_rejections": 100
    }
}
```

## Model Files

The server requires the following model files:

- **MTCNN Models** (in `src/dnn/models/`):
  - `det1.caffemodel` & `det1.prototxt` - P-Net
  - `det2.caffemodel` & `det2.prototxt` - R-Net  
  - `det3.caffemodel` & `det3.prototxt` - O-Net

- **Face Embedding Model** (in `models/`):
  - `inception.onnx` - Inception ResNet v1 for face embeddings

## Performance

- **Processing Speed**: ~20-30 FPS on modern hardware
- **Memory Usage**: ~2-4GB RAM (depending on batch size)
- **GPU Acceleration**: Supported via CUDA (2-3x speedup)
- **Concurrent Connections**: Supports multiple WebSocket clients

## Troubleshooting

### Common Issues

1. **ONNX Runtime not found**:
   ```bash
   export LD_LIBRARY_PATH=/usr/local/onnxruntime/lib:$LD_LIBRARY_PATH
   ```

2. **PostgreSQL connection failed**:
   - Check database credentials in `config.ini`
   - Ensure PostgreSQL is running: `sudo systemctl status postgresql`

3. **OpenCV conflicts with conda**:
   - The CMakeLists.txt automatically excludes conda paths
   - Use system OpenCV: `sudo apt install libopencv-dev`

4. **CUDA not detected**:
   - Install CUDA toolkit and drivers
   - Verify with: `nvidia-smi`

## Development

### Building Tests

```bash
cd build
make test_mtcnn test_onnx
./test_mtcnn
./test_onnx
```

### Code Structure

```
src/
├── dnn/           # MTCNN face detection implementation
├── onnx/          # ONNX Runtime utilities
├── postgres/      # Database integration
├── websocket/     # WebSocket server components
└── test/          # Unit tests
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## Support

For issues and questions:
- Create an issue on GitHub
- Check the troubleshooting section
- Review the configuration options