from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional
import io
import os
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForImageClassification
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="SANTUN Messaging Apps API",
    description="API for text rewriting and NSFW image detection",
    version="1.0.0"
)

# Initialize LLM with API key from environment variables
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY environment variable is not set. Please check your .env file.")

llm = ChatGoogleGenerativeAI(model='gemini-2.0-flash-exp', google_api_key=api_key)

# Load NSFW detection model (lazy loading - will only load when endpoint is called)
processor = None
model = None

def load_nsfw_model():
    global processor, model
    if processor is None or model is None:
        processor = AutoProcessor.from_pretrained("Falconsai/nsfw_image_detection")
        model = AutoModelForImageClassification.from_pretrained("Falconsai/nsfw_image_detection")
    return processor, model

# Pydantic models for request and response validation
class TextRequest(BaseModel):
    text: str = Field(..., description="The text to analyze and rewrite if necessary")

class TextResponse(BaseModel):
    classification: str = Field(..., description="Classification result: 'kasar', 'gaul', or 'netral'")
    rewritten_text: str = Field(..., description="The rewritten, more polite version of the text")

class NSFWResponse(BaseModel):
    nsfw_score: float = Field(..., description="Confidence score for NSFW content (0-1)")
    is_nsfw: bool = Field(..., description="Whether the image is classified as NSFW")

# Text classification function
def classify_text(text: str) -> str:
    """Classify text as 'kasar', 'gaul', or 'netral'."""
    prompt = f"""
    Classify the following Indonesian text into one of these categories: 'kasar', 'gaul', or 'netral'.
    Text: "{text}"
    Return ONLY ONE of these exact words: 'kasar', 'gaul', or 'netral'.
    """
    response = llm.invoke(prompt)
    # Extract and normalize classification
    classification = response.content.strip().lower()
    # Ensure we only return one of the expected values
    if classification not in ["kasar", "gaul", "netral"]:
        classification = "netral"  # Default to netral if unexpected response
    return classification

# Text rewriting function
def rewrite_text(text: str, classification: str) -> str:
    """Rewrite text to be more polite based on classification."""
    if classification == "netral":
        return text  # No need to rewrite neutral text
    
    prompt_template = """
    Berikut adalah teks dalam Bahasa Indonesia: "{text}"
    Teks ini terklasifikasi sebagai "{classification}".
    
    Tugas Anda adalah untuk menulis ulang teks tersebut menjadi lebih sopan dan formal,
    namun tetap mempertahankan maksud aslinya.
    
    Jika teks mengandung kata-kata kasar, ganti dengan kata-kata yang lebih sopan.
    Jika teks menggunakan bahasa gaul, ubah menjadi bahasa formal yang baik dan benar.
    
    Berikan HANYA hasil penulisan ulang saja tanpa penjelasan tambahan.
    """
    
    prompt = PromptTemplate(template=prompt_template, input_variables=["text", "classification"])
    response = llm.invoke(prompt.invoke({"text": text, "classification": classification}))
    return response.content.strip()

@app.post("/rewrite", response_model=TextResponse)
async def rewrite_endpoint(request: TextRequest):
    """
    Analyzes text and rewrites it to be more polite if necessary.
    
    - **text**: The input text to analyze and potentially rewrite
    
    Returns:
    - **classification**: Whether the text is 'kasar', 'gaul', or 'netral'
    - **rewritten_text**: The polite version of the text
    """
    try:
        # Get the classification
        classification = classify_text(request.text)
        
        # Rewrite the text if needed
        rewritten = rewrite_text(request.text, classification)
        
        return TextResponse(
            classification=classification,
            rewritten_text=rewritten
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing text: {str(e)}")

@app.post("/detect-nsfw", response_model=NSFWResponse)
async def detect_nsfw(file: UploadFile = File(...)):
    """
    Detects if an uploaded image contains NSFW content.
    
    - **file**: The image file to analyze
    
    Returns:
    - **nsfw_score**: Confidence score for NSFW content (0-1)
    - **is_nsfw**: Boolean indicating if the image is classified as NSFW
    """
    try:
        # Read and process the image
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        # Load the model if not already loaded
        processor, model = load_nsfw_model()
        
        # Prepare input and predict
        inputs = processor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=1)
        nsfw_index = 1  # Assuming index 1 corresponds to NSFW class
        nsfw_score = probabilities[0, nsfw_index].item()
        is_nsfw = nsfw_score > 0.5  # Threshold can be adjusted
        
        return NSFWResponse(
            nsfw_score=nsfw_score,
            is_nsfw=is_nsfw
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

# Root endpoint for basic API info
@app.get("/")
async def root():
    return {
        "message": "Welcome to SANTUN Messaging Apps API",
        "endpoints": {
            "/rewrite": "Text classification and rewriting",
            "/detect-nsfw": "NSFW image detection"
        }
    }

# Run the application with: uvicorn main:app --reload
if __name__ == "__main__":
    import uvicorn
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    debug = os.getenv("DEBUG", "False").lower() in ("true", "1", "t")
    uvicorn.run("main:app", host=host, port=port, reload=debug)
