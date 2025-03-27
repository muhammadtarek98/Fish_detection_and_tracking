from fastapi import FastAPI, UploadFile, File, Request, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
import os,shutil
import torch
from tracker import *
import uvicorn

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"])

templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("upload.html", {"request": request})


@app.post("/process_video/")
async def process_video_endpoint(file: UploadFile = File(...)):
    input_path = f"temp_{file.filename}"
    output_path = f"output_{file.filename}"

    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = get_the_model(ckpt_path="/home/muhammad/projects/SEE_assessment/detection_model.ckpt",device= device)
    tracker = load_tracker()

    try:
        summary = process_video(input_path, output_path, model, tracker, device)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(input_path):
            os.remove(input_path)

    return JSONResponse(
        content={"message": "Video processed successfully", "output_path": output_path, "summary": summary})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)