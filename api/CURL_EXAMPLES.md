# Face Recognition API - cURL Examples

## Server Configuration
- Default host: `localhost`
- Default port: `7999`
- Base URL: `http://localhost:7999`

You can override these by setting environment variables:
```bash
export API_HOST=0.0.0.0
export API_PORT=8000
```

## Endpoints

### 1. Health Check (GET)
```bash
curl -X GET http://localhost:7999/health \
  -H "Content-Type: application/json"
```

**Response:**
```json
{
  "status": "healthy",
  "recognizer_initialized": true
}
```

### 2. Root Endpoint (GET)
```bash
curl -X GET http://localhost:7999/ \
  -H "Content-Type: application/json"
```

**Response:**
```json
{
  "message": "Face Recognition API",
  "version": "1.0.0",
  "endpoints": {
    "recognize": "/recognize",
    "register": "/register",
    "health": "/health"
  }
}
```

### 3. Recognize Face (POST)

Recognize faces in an image. Supports both JSON (base64) and multipart/form-data image uploads.

**Option A: JSON with base64 image**
```bash
curl -X POST http://localhost:7999/recognize \
  -H "Content-Type: application/json" \
  -d '{
    "base64_image": "<base64_encoded_image>"
  }'
```

**Option B: Multipart form data with image file**
```bash
curl --location 'http://localhost:7999/recognize' \
  --header 'Content-Type: image/jpeg' \
  --form 'file=@"/path/to/image.jpg"'
```

**Examples with image file:**

JSON (base64):
```bash
# Encode image to base64
IMAGE_BASE64=$(base64 -w 0 /path/to/image.jpg)

# Send request
curl -X POST http://localhost:7999/recognize \
  -H "Content-Type: application/json" \
  -d "{\"base64_image\": \"${IMAGE_BASE64}\"}"
```

Multipart form data:
```bash
curl --location 'http://localhost:7999/recognize' \
  --header 'Content-Type: image/jpeg' \
  --form 'file=@"/path/to/image.jpg"'
```

**One-liners:**
```bash
# JSON (base64)
curl -X POST http://localhost:7999/recognize \
  -H "Content-Type: application/json" \
  -d "{\"base64_image\": \"$(base64 -w 0 /path/to/image.jpg)\"}"

# Multipart form data
curl --location 'http://localhost:7999/recognize' \
  --header 'Content-Type: image/jpeg' \
  --form 'file=@"/path/to/image.jpg"'
```

**Response:**
```json
{
  "faces": [
    {
      "name": "John Doe",
      "confidence": 0.95,
      "status": "recognized",
      "xmin": 100.5,
      "ymin": 150.2,
      "xmax": 250.8,
      "ymax": 300.5,
      "person_id": "d19854ef-d2eb-4e80-94be-be82444783e9"
    }
  ],
  "total_faces": 1
}
```

### 4. Register Face (POST)

Register a new face with a person name. Supports both JSON (base64) and multipart/form-data image uploads.

**Note:** The `name` field is **required** and cannot be empty.

**Option A: JSON with base64 image**
```bash
curl -X POST http://localhost:7999/register \
  -H "Content-Type: application/json" \
  -d '{
    "name": "John Doe",
    "base64_image": "<base64_encoded_image>"
  }'
```

**Option B: Multipart form data with image file (name in form field)**
```bash
curl --location 'http://localhost:7999/register' \
  --header 'Content-Type: image/jpeg' \
  --form 'name="John Doe"' \
  --form 'file=@"/path/to/image.jpg"'
```

**Option C: Multipart form data with image file (name in query parameter)**
```bash
curl --location 'http://localhost:7999/register?name=John%20Doe' \
  --header 'Content-Type: image/jpeg' \
  --form 'file=@"/path/to/image.jpg"'
```

**Examples with image file:**

JSON (base64):
```bash
# Encode image to base64
IMAGE_BASE64=$(base64 -w 0 /path/to/image.jpg)

# Send request with person name
curl -X POST http://localhost:7999/register \
  -H "Content-Type: application/json" \
  -d "{\"name\": \"John Doe\", \"base64_image\": \"${IMAGE_BASE64}\"}"
```

Multipart form data (name in form):
```bash
curl --location 'http://localhost:7999/register' \
  --header 'Content-Type: image/jpeg' \
  --form 'name="John Doe"' \
  --form 'file=@"/path/to/image.jpg"'
```

Multipart form data (name in query):
```bash
curl --location 'http://localhost:7999/register?name=John%20Doe' \
  --header 'Content-Type: image/jpeg' \
  --form 'file=@"/path/to/image.jpg"'
```

**One-liners:**
```bash
# JSON (base64)
curl -X POST http://localhost:7999/register \
  -H "Content-Type: application/json" \
  -d "{\"name\": \"John Doe\", \"base64_image\": \"$(base64 -w 0 /path/to/image.jpg)\"}"

# Multipart form data (name in form)
curl --location 'http://localhost:7999/register' \
  --header 'Content-Type: image/jpeg' \
  --form 'name="John Doe"' \
  --form 'file=@"/path/to/image.jpg"'

# Multipart form data (name in query)
curl --location 'http://localhost:7999/register?name=John%20Doe' \
  --header 'Content-Type: image/jpeg' \
  --form 'file=@"/path/to/image.jpg"'
```

**Response:**
```json
{
  "success": true,
  "message": "Face registered successfully for name: John Doe: Success message"
}
```

## Helper Script

Use the provided `curl_examples.sh` script for easier testing:

```bash
# Make executable
chmod +x curl_examples.sh

# Run with an image file
./curl_examples.sh /path/to/image.jpg

# Run with image file and person name for register
./curl_examples.sh /path/to/image.jpg "John Doe"
```

## Notes

- **Image Upload Methods:**
  - **JSON with base64**: Send image as base64-encoded string in JSON body
  - **Multipart form data**: Send image file using `--form 'file=@"<path>"'` (recommended)
  
- **Content-Type Headers:**
  - For JSON: `Content-Type: application/json`
  - For multipart: `Content-Type: image/jpeg` (or `image/png`, `multipart/form-data` - curl will set this automatically)
  
- **Register Endpoint:**
  - For JSON: Include `name` field in JSON body (string, **required**)
  - For multipart: Include `name` as form field `--form 'name="<person_name>"'` or as query parameter `/register?name=<person_name>`
  - The `name` field is **required** and cannot be empty
  
- **Supported image formats**: JPEG, PNG, etc. (depends on OpenCV support)
- **Image format**: The API expects images in BGR format (OpenCV standard)
- **Auto-detection**: If Content-Type is not specified, the API will try both binary and JSON parsing

## Error Responses

**400 Bad Request:**
```json
"Either image file or base64_image must be provided"
```

**400 Bad Request:**
```json
"Failed to decode image"
```

**400 Bad Request:**
```json
"Name field must be provided and cannot be empty"
```

**500 Internal Server Error:**
```json
"Error processing image: <error_message>"
```

**500 Internal Server Error:**
```json
"Error registering face: <error_message>"
```

