#!/bin/bash

# Face Recognition API - cURL Examples
# Default server: http://localhost:7999
# Set API_HOST and API_PORT environment variables to override

API_HOST=${API_HOST:-localhost}
API_PORT=${API_PORT:-7999}
BASE_URL="http://${API_HOST}:${API_PORT}"

echo "Face Recognition API cURL Examples"
echo "=================================="
echo ""

# Helper function to encode image to base64
encode_image() {
    if [ -z "$1" ]; then
        echo "Usage: encode_image <image_file_path>"
        return 1
    fi
    if [ ! -f "$1" ]; then
        echo "Error: File $1 not found"
        return 1
    fi
    base64 -w 0 "$1"
}

# Example 1: Health Check (GET)
echo "1. Health Check:"
echo "curl -X GET ${BASE_URL}/health"
echo ""
curl -X GET "${BASE_URL}/health" -H "Content-Type: application/json" | jq .
echo ""
echo "---"
echo ""

# Example 2: Root endpoint (GET)
echo "2. Root endpoint:"
echo "curl -X GET ${BASE_URL}/"
echo ""
curl -X GET "${BASE_URL}/" -H "Content-Type: application/json" | jq .
echo ""
echo "---"
echo ""

# Example 3: Recognize face (POST)
# Supports both JSON (base64) and multipart form data image uploads
echo "3. Recognize face:"
echo ""
echo "Option A: JSON with base64 image:"
echo "curl -X POST ${BASE_URL}/recognize \\"
echo "  -H 'Content-Type: application/json' \\"
echo "  -d '{\"base64_image\": \"<base64_encoded_image>\"}'"
echo ""
echo "Option B: Multipart form data with image file:"
echo "curl --location '${BASE_URL}/recognize' \\"
echo "  --form 'file=@\"/path/to/image.jpg\"'"
echo ""
if [ -n "$1" ] && [ -f "$1" ]; then
    echo "Testing with image file: $1"
    echo ""
    echo "A) JSON (base64):"
    IMAGE_BASE64=$(encode_image "$1")
    curl -X POST "${BASE_URL}/recognize" \
        -H "Content-Type: application/json" \
        -d "{\"base64_image\": \"${IMAGE_BASE64}\"}" | jq .
    echo ""
    echo "B) Multipart form data:"
    curl --location "${BASE_URL}/recognize" \
        --form "file=@\"$1\"" | jq .
else
    echo "  # To test with an image file, run:"
    echo "  # ./curl_examples.sh /path/to/image.jpg"
    echo ""
    echo "  # JSON (base64) example:"
    echo "  IMAGE_BASE64=\$(base64 -w 0 /path/to/image.jpg)"
    echo "  curl -X POST ${BASE_URL}/recognize \\"
    echo "    -H 'Content-Type: application/json' \\"
    echo "    -d \"{\\\"base64_image\\\": \\\"\${IMAGE_BASE64}\\\"}\""
    echo ""
  echo "  # Multipart form data example:"
  echo "  curl --location '${BASE_URL}/recognize' \\"
  echo "    --form 'file=@\"/path/to/image.jpg\"'"
fi
echo ""
echo "---"
echo ""

# Example 4: Register face (POST)
# Supports both JSON (base64) and multipart form data image uploads
# Note: For multipart uploads, name can be provided in form field or query parameter
# Note: The name field is required and cannot be empty
echo "4. Register face:"
echo ""
echo "Option A: JSON with base64 image:"
echo "curl -X POST ${BASE_URL}/register \\"
echo "  -H 'Content-Type: application/json' \\"
echo "  -d '{\"name\": \"<person_name>\", \"base64_image\": \"<base64_encoded_image>\"}'"
echo ""
echo "Option B: Multipart form data (name in form field):"
echo "curl --location '${BASE_URL}/register' \\"
echo "  --form 'name=\"<person_name>\"' \\"
echo "  --form 'file=@\"/path/to/image.jpg\"'"
echo ""
echo "Option C: Multipart form data (name in query parameter):"
echo "curl --location '${BASE_URL}/register?name=<person_name>' \\"
echo "  --form 'file=@\"/path/to/image.jpg\"'"
echo ""
if [ -n "$1" ] && [ -f "$1" ]; then
    echo "Testing with image file: $1"
    PERSON_NAME=${2:-"John Doe"}
    echo ""
    echo "A) JSON (base64):"
    IMAGE_BASE64=$(encode_image "$1")
    curl -X POST "${BASE_URL}/register" \
        -H "Content-Type: application/json" \
        -d "{\"name\": \"${PERSON_NAME}\", \"base64_image\": \"${IMAGE_BASE64}\"}" | jq .
    echo ""
    echo "B) Multipart form data (name in form):"
    curl --location "${BASE_URL}/register" \
        --form "name=\"${PERSON_NAME}\"" \
        --form "file=@\"$1\"" | jq .
    echo ""
    echo "C) Multipart form data (name in query):"
    # URL encode the name for query parameter
    ENCODED_NAME=$(echo "$PERSON_NAME" | sed 's/ /%20/g')
    curl --location "${BASE_URL}/register?name=${ENCODED_NAME}" \
        --form "file=@\"$1\"" | jq .
else
    echo "  # To test with an image file, run:"
    echo "  # ./curl_examples.sh /path/to/image.jpg [person_name]"
    echo ""
    echo "  # JSON (base64) example:"
    echo "  IMAGE_BASE64=\$(base64 -w 0 /path/to/image.jpg)"
    echo "  curl -X POST ${BASE_URL}/register \\"
    echo "    -H 'Content-Type: application/json' \\"
    echo "    -d \"{\\\"name\\\": \\\"John Doe\\\", \\\"base64_image\\\": \\\"\${IMAGE_BASE64}\\\"}\""
    echo ""
  echo "  # Multipart form data example (name in form):"
  echo "  curl --location '${BASE_URL}/register' \\"
  echo "    --form 'name=\"John Doe\"' \\"
  echo "    --form 'file=@\"/path/to/image.jpg\"'"
    echo ""
  echo "  # Multipart form data example (name in query):"
  echo "  curl --location '${BASE_URL}/register?name=John%20Doe' \\"
  echo "    --form 'file=@\"/path/to/image.jpg\"'"
fi
echo ""

