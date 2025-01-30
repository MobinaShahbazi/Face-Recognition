from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import shutil
import os
from fastapi import Request

app = FastAPI()

# Define paths for templates and static files
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Serve the upload form page
@app.get("/", response_class=HTMLResponse)
async def get_upload_form(request: Request):  # Ensure `request` is included as a parameter
    return templates.TemplateResponse("upload_form.html", {"request": request, "message": None})

# Handle the file upload
@app.post("/upload/", response_class=HTMLResponse)
async def upload_image(request: Request, file: UploadFile = File(...)):
    try:
        # Save the uploaded file
        file_location = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_location, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Return a success message to the user
        message = f"Upload successful! File saved at {file_location}"
        return templates.TemplateResponse("upload_form.html", {"request": request, "message": message})

    except Exception as e:
        message = f"Upload failed: {str(e)}"
        return templates.TemplateResponse("upload_form.html", {"request": request, "message": message})
