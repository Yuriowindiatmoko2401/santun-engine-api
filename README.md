# SANTUN Messaging Apps API

A FastAPI application providing endpoints for text classification/rewriting and NSFW image detection.

## Features

- **Text Classification and Rewriting**: Analyzes Indonesian text and rewrites inappropriate content to be more polite
- **NSFW Image Detection**: Detects whether uploaded images contain inappropriate content

## Getting Started

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Installation

1. Clone this repository or download the source code
2. Create a `.env` file in the root directory with your API keys:

```
GOOGLE_API_KEY=your_google_api_key_here
# Optional configuration
# PORT=8000
# HOST=0.0.0.0
# DEBUG=True
```

3. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Running the API Server

Start the FastAPI server with:

```bash
uvicorn main:app --reload
```

The API will be available at http://127.0.0.1:8000/

Interactive API documentation: http://127.0.0.1:8000/docs

### API Endpoints

#### 1. Text Rewriting

- **Endpoint**: `POST /rewrite`
- **Input**:
```json
{
  "text": "input kalimat tidak sopan"
}
```
- **Output**:
```json
{
  "classification": "kasar",
  "rewritten_texts": ["kalimat yang telah disunting lebih santun", "kalimat yang telah disunting lebih santun", "kalimat yang telah disunting lebih santun", "kalimat yang telah disunting lebih santun", "kalimat yang telah disunting lebih santun"]
}
```

#### 2. NSFW Image Detection

- **Endpoint**: `POST /detect-nsfw`
- **Input**: Form-data with an image file
- **Output**:
```json
{
  "nsfw_score": 0.89,
  "is_nsfw": true
}
```

## Example Usage

### Text Rewriting

```python
import requests

url = "http://127.0.0.1:8000/rewrite"
payload = {"text": "anjing kamu ini"}
response = requests.post(url, json=payload)
print(response.json())
```

### NSFW Image Detection

```python
import requests

url = "http://127.0.0.1:8000/detect-nsfw"
files = {"file": open("path/to/image.jpg", "rb")}
response = requests.post(url, files=files)
print(response.json())
```

## Using in Google Colab

To use this API in Google Colab, follow these steps:

1. Upload all files to Colab
2. Install dependencies:
   ```
   !pip install -r requirements.txt
   ```
3. Run the FastAPI server with ngrok to expose it:
   ```python
   !pip install pyngrok
   from pyngrok import ngrok
   
   # Start the server in the background
   !uvicorn main:app --port 8000 &
   
   # Create a tunnel to the server
   public_url = ngrok.connect(8000)
   print(f"Public URL: {public_url}")
   ```
4. Use the public URL to access the API endpoints

## License

[MIT License](LICENSE)

## Acknowledgments

This project uses:
- FastAPI for API framework
- Langchain with Google Generative AI for text processing
- Transformers library with Falconsai's NSFW image detection model
