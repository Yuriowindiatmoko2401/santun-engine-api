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

# Load NSFW detection model (lazy loading - will only load when endpoint is called) - placeholder global
processor = None
model = None

"""
NSFW detector sudah didapat opsi model lokal terbaik yaitu Telah dilakukan evaluasi model NSFW detector, dan 
diperoleh dua opsi model 
- ("AdamCodd/vit-base-nsfw-detector”) dan 
- ("openai/clip-vit-large-patch14”) - img-3 , 4, 5 :
"""

def load_nsfw_model():
    """
    Loads the NSFW detection model.
    Returns processor and model for image classification.
    Includes verbose error handling to diagnose import issues.
    """
    global processor, model
    
    print("Starting to load NSFW model...")
    
    if processor is not None and model is not None:
        print("Model already loaded, returning cached version")
        return processor, model
    
    try:
        print("Importing necessary components...")
        # Try explicit imports first to diagnose any issues
        import transformers
        print(f"Transformers version: {transformers.__version__}")
        
        import huggingface_hub
        print(f"Huggingface Hub version: {huggingface_hub.__version__}")
        
        # Check if the required function exists
        try:
            from huggingface_hub import split_torch_state_dict_into_shards
            print("Successfully imported split_torch_state_dict_into_shards")
        except ImportError:
            print("WARNING: split_torch_state_dict_into_shards not found in huggingface_hub")
            print("This may cause issues with the transformers library")
        
        print("Loading processor from AdamCodd/vit-base-nsfw-detector")
        processor = AutoProcessor.from_pretrained("AdamCodd/vit-base-nsfw-detector")
        print("Processor loaded successfully")
        
        print("Loading model from AdamCodd/vit-base-nsfw-detector")
        model = AutoModelForImageClassification.from_pretrained("AdamCodd/vit-base-nsfw-detector")
        print("Model loaded successfully")
        
        return processor, model
    
    except ImportError as e:
        print(f"Import error: {str(e)}")
        import traceback
        print(traceback.format_exc())
        
        # Try alternative imports as a fallback
        try:
            print("Attempting alternative approach...")
            
            # Try different model as fallback
            print("Trying fallback model...")
            from transformers import ViTImageProcessor, ViTForImageClassification
            
            print("Loading processor using direct class import")
            processor = ViTImageProcessor.from_pretrained("AdamCodd/vit-base-nsfw-detector")
            print("Processor loaded successfully")
            
            print("Loading model using direct class import")
            model = ViTForImageClassification.from_pretrained("AdamCodd/vit-base-nsfw-detector")
            print("Model loaded successfully")
            
            return processor, model
        
        except Exception as fallback_error:
            print(f"Fallback approach also failed: {str(fallback_error)}")
            import traceback
            print(traceback.format_exc())
            raise RuntimeError(f"Failed to load NSFW model: {str(e)} -> Fallback error: {str(fallback_error)}")
    
    except Exception as e:
        print(f"Unexpected error loading NSFW model: {str(e)}")
        import traceback
        print(traceback.format_exc())
        raise

# Pydantic models for request and response validation
class TextRequest(BaseModel):
    text: str = Field(..., description="The text to analyze and rewrite if necessary")

class TextResponse(BaseModel):
    classification: str = Field(..., description="Classification result: 'kasar', 'gaul', or 'netral'")
    rewritten_texts: List[str] = Field(..., description="List of rewritten, more polite versions of the text")

class NSFWResponse(BaseModel):
    nsfw_score: float = Field(..., description="Confidence score for NSFW content (0-1)")
    is_nsfw: bool = Field(..., description="Whether the image is classified as NSFW")

# Text classification function --disini
def classify_text(text: str) -> str:
    """Classify text as 'kasar', 'gaul', or 'netral'."""
    prompt = f"""
    Classify the following Indonesian text (which may contain slang, regional languages, or mixed languages) into one of these categories:
    - 'kasar' (vulgar/impolite language, swear words, or offensive terms)
    - 'gaul' (slang, informal language, or mixed language including English/regional terms)
    - 'netral' (standard/formal Indonesian without offensive or slang elements)

    Text to classify: "{text}"

    Consider these guidelines:
    1. If the text contains swear words, offensive terms, or vulgar language, classify as 'kasar'
    2. If the text uses informal language, slang, or mixes Indonesian with English/regional languages but isn't offensive, classify as 'gaul'
    3. If the text is in standard/formal Indonesian without slang or offensive terms, classify as 'netral'

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
def rewrite_text(text: str, classification: str) -> List[str]:
    """Rewrite text to generate multiple polite versions based on classification."""
    if classification == "netral":
        return [text]  # No need to rewrite neutral text
    
    prompt_template = """
    Berikut adalah teks dalam Bahasa Indonesia: "{text}"
    Teks ini terklasifikasi sebagai "{classification}".
    
    Tugas Anda adalah untuk menulis ulang teks tersebut menjadi 5 versi yang lebih sopan dan formal,
    namun tetap mempertahankan maksud aslinya.
    
    Jika teks mengandung kata-kata kasar, ganti dengan kata-kata yang lebih sopan.
    Jika teks menggunakan bahasa gaul, ubah menjadi bahasa formal yang baik dan benar.
    
    Format output:
    1. [versi pertama]
    2. [versi kedua]
    3. [versi ketiga]
    4. [versi keempat]
    5. [versi kelima]
    
    Berikan HANYA hasil penulisan ulang saja tanpa penjelasan tambahan.
    """
    
    prompt = PromptTemplate(template=prompt_template, input_variables=["text", "classification"])
    response = llm.invoke(prompt.invoke({"text": text, "classification": classification}))
    
    # Parse the response to extract the 5 versions
    response_text = response.content.strip()
    versions = []
    
    # Try to extract numbered versions (1. version, 2. version, etc.)
    import re
    numbered_versions = re.findall(r'^\d+\.\s+(.*?)$', response_text, re.MULTILINE)
    
    if len(numbered_versions) >= 5:
        # If we successfully extracted at least 5 numbered versions
        versions = numbered_versions[:5]
    else:
        # Fallback: split by newlines and try to get 5 versions
        lines = [line.strip() for line in response_text.split('\n') if line.strip()]
        versions = lines[:5]
    
    # If we still don't have 5 versions, pad with the first version
    while len(versions) < 5:
        if versions:
            versions.append(versions[0])
        else:
            versions.append(text)  # Fallback to original text if nothing was extracted
    
    return versions

@app.post("/rewrite", response_model=TextResponse)
async def rewrite_endpoint(request: TextRequest):
    """
    Analyzes text and rewrites it to be more polite if necessary.
    
    - **text**: The input text to analyze and potentially rewrite
    
    Returns:
    - **classification**: Whether the text is 'kasar', 'gaul', or 'netral'
    - **rewritten_texts**: List of rewritten, more polite versions of the text
    """
    try:
        # Get the classification
        classification = classify_text(request.text)
        
        # Rewrite the text if needed
        rewritten_versions = rewrite_text(request.text, classification)
        
        return TextResponse(
            classification=classification,
            rewritten_texts=rewritten_versions
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
    print("Starting NSFW detection process")
    try:
        # Read and process the image
        print("Reading uploaded file")
        image_bytes = await file.read()
        print(f"File read successfully, size: {len(image_bytes)} bytes")
        
        print("Opening image with PIL")
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        print(f"Image opened successfully, size: {image.size}")
        
        # Load the model if not already loaded
        print("Loading NSFW detection model")
        try:
            processor, model = load_nsfw_model()
            print("Model and processor loaded successfully")
        except Exception as model_error:
            print(f"Error loading model: {str(model_error)}")
            import traceback
            print(traceback.format_exc())
            raise
        
        # Prepare input and predict
        print("Preparing input for model")
        try:
            inputs = processor(images=image, return_tensors="pt")
            print(f"Input prepared successfully, shape: {inputs['pixel_values'].shape}")
        except Exception as proc_error:
            print(f"Error during processing: {str(proc_error)}")
            import traceback
            print(traceback.format_exc())
            raise
        
        print("Running model inference")
        try:
            with torch.no_grad():
                outputs = model(**inputs)
            print("Model inference completed successfully")
        except Exception as inf_error:
            print(f"Error during inference: {str(inf_error)}")
            import traceback
            print(traceback.format_exc())
            raise
        
        print("Processing model outputs")
        logits = outputs.logits
        print(f"Logits shape: {logits.shape}")
        probabilities = torch.nn.functional.softmax(logits, dim=1)
        print(f"Probabilities calculated, shape: {probabilities.shape}")
        nsfw_index = 1  # Assuming index 1 corresponds to NSFW class
        nsfw_score = probabilities[0, nsfw_index].item()
        print(f"NSFW score: {nsfw_score}")
        is_nsfw = nsfw_score > 0.5  # Threshold can be adjusted
        print(f"Final classification - is_nsfw: {is_nsfw}")
        
        return NSFWResponse(
            nsfw_score=nsfw_score,
            is_nsfw=is_nsfw
        )
    except Exception as e:
        print(f"Error in NSFW detection: {str(e)}")
        import traceback
        print(traceback.format_exc())
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
