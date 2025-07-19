from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
from inference.text import TextPredictor
from inference.dictionary import DictionaryChecker
from replace import replace_text
# from inference.image import model_loader, predictor # Uncomment and implement when image processing is ready

app = FastAPI(
    title="Kitty AI Server",
    description="AI-powered digital safe communication platform for youth",
    version="0.1.0"
)

# Initialize models (assuming they are already initialized in replace.py, but good to have here for clarity)
# These are already initialized in replace.py, so we can directly use replace_text
# checker = DictionaryChecker("inference/dictionary/dictionary.csv")
# txt_pred = TextPredictor("inference/text/model")

class TextProcessRequest(BaseModel):
    text: str
    model: str = "gpt-4o-mini"
    show_prompt: bool = False

class ImageProcessRequest(BaseModel):
    image_path: str # Or base64 encoded image string

@app.post("/process_text")
async def process_text_endpoint(request: TextProcessRequest):
    try:
        result = replace_text(request.text, model=request.model, show_prompt=request.show_prompt)
        return {"original_text": request.text, "processed_text": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/process_image")
async def process_image_endpoint(request: ImageProcessRequest):
    # Placeholder for image processing
    # You'll need to load your image model and implement the prediction logic here.
    # Example:
    # try:
    #     image_model = model_loader.load_model("inference/image/model_ts.pt")
    #     prediction = predictor.predict(image_model, request.image_path)
    #     return {"image_path": request.image_path, "prediction": prediction}
    # except Exception as e:
    #     raise HTTPException(status_code=500, detail=str(e))
    return {"message": "Image processing endpoint is not yet implemented.", "received_image_path": request.image_path}

@app.get("/")
async def root():
    return {"message": "Kitty AI Server is running!"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
