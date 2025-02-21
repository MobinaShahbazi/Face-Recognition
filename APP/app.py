from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import os
import shutil

from utils import FaceRecognition

app = FastAPI()
image_folder = 'registered'
face_recog = FaceRecognition()
face_recog.register_all(image_folder)

# Define paths
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.get("/", response_class=HTMLResponse)
async def get_upload_form(request: Request):
    return templates.TemplateResponse("upload_form.html", {"request": request, "message": None})

@app.post("/upload/", response_class=HTMLResponse)
async def upload_image(request: Request, file: UploadFile = File(...)):
    try:
        file_location = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_location, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)


        test_image_path = file_location
        message = face_recog.authenticate(test_image_path)
        return templates.TemplateResponse("upload_form.html", {"request": request, "message": message, "filename": file.filename})

    except Exception as e:
        return templates.TemplateResponse("upload_form.html", {"request": request, "message": f"Upload failed: {str(e)}"})
