# Body Shape Prediction API

This Flask-based API provides endpoints for body shape prediction from images.

## Setup

### Prerequisites

- Python 3.8+
- pip

### Installation

1. Clone the repository (if you haven't already)
2. Navigate to the project directory
3. Install the requirements:

```bash
pip install -r api/requirements.txt
```

### Running the API

From the project root directory, run:

```bash
python api/app.py
```

The API will start on http://localhost:5000

## API Endpoints

### Health Check

- **URL**: `/health`
- **Method**: `GET`
- **Response**: `{"status": "ok", "message": "Body Shape Prediction API is running"}`

### Body Shape Prediction

- **URL**: `/predict`
- **Method**: `POST`
- **Form Parameters**:
  - `image`: The image file (required)
  - `height`: Person's height in cm (required)
  - `is_female`: Boolean value (true/false), defaults to true
  - `debug`: Boolean value (true/false), defaults to false
  - `force_shape`: Force a specific shape classification (optional)
  - `skip_correction`: Skip silhouette-based hip correction (optional)

- **Response Format**:
```json
{
  "shape": "Shape name",
  "confidence": 95.5,
  "description": "Description of the body shape",
  "result_image_url": "/results/filename.jpg",
  "fashion_pose": false,
  "is_female": true
}
```

- **Debug Response** (When debug=true, includes additional feature data):
```json
{
  "shape": "Shape name",
  "confidence": 95.5,
  "description": "Description of the body shape",
  "result_image_url": "/results/filename.jpg",
  "fashion_pose": false,
  "is_female": true,
  "features": {
    "waist_to_hip_ratio": 0.8,
    "shoulder_to_waist_ratio": 1.2,
    ...
  }
}
```

### Accessing Result Images

- **URL**: `/results/<filename>`
- **Method**: `GET`
- **Response**: The image file

## Testing the API

Use the included test script to test the API:

```bash
python api/test_api.py --image path/to/image.jpg --height 170
```

### Test Script Options

- `--url`: Base URL of the API (default: http://localhost:5000)
- `--image`: Path to input image (required)
- `--height`: Person height in cm (required)
- `--female`: Flag to specify female (default: true)
- `--debug`: Enable debug mode

## Example

```bash
python api/test_api.py --image samples/person.jpg --height 170 --debug
```

## Error Handling

The API returns appropriate HTTP status codes:
- 200: Success
- 400: Client error (wrong input)
- 500: Server error

Error responses include an error message:
```json
{
  "error": "Description of the error"
}
``` 