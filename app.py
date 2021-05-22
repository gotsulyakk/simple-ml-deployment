import cv2
import uvicorn
import numpy as np
from fastapi import FastAPI, Request
from fastapi import UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from core import utils
from core.model import YOLOModel


app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

model = YOLOModel(weights_path="./model")


@app.get("/home", response_class=HTMLResponse)
def render_home(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})


@app.get("/predict", response_class=HTMLResponse)
@app.post("/predict")
async def predict(request: Request,
                  image_file: UploadFile = File(...),
                  iou_threshold: float = 0.45,
                  score_threshold: float = 0.25
                  ):
    image = utils.load_imagefile(await image_file.read())
    image_data = utils.preprocess_image_pil(image)

    predictions = model.predict(
        image_data, iou_threshold=iou_threshold, score_threshold=score_threshold
        )

    image_arr = np.asarray(image)
    image_viz = utils.draw_bbox(image_arr, predictions)
    image_viz = cv2.cvtColor(image_viz, cv2.COLOR_BGR2RGB)
    utils.save_image(image_viz)

    num_detections = utils.count_detections(predictions)
    for key, value in num_detections.items():
        print(key, len([item for item in value if item]))

    return templates.TemplateResponse("result.html", {"request": request})
    
    
if __name__ == "__main__":
    uvicorn.run(app, port=8000, host="0.0.0.0")
