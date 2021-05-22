import pickle
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from core import utils
from core.model import YOLOModel

# 1. Load image
# 2. Preprocess image
# 3. Load model
# 4. Get predictions
# 5. Print/Visualize predictions

IMAGE_PATH = "./data/images/tryhiv.png"
WEIGHTS_PATH = "./model"

INPUT_SIZE = 416
IOU_THRESHOLD = 0.45
SCORE_THRESHOLD = 0.25

original_image = utils.load_image(image_path=IMAGE_PATH)
image_data = utils.preprocess_image(original_image, input_size=INPUT_SIZE)

model = YOLOModel(weights_path=WEIGHTS_PATH)

predictions = model.predict(image_data, iou_threshold=IOU_THRESHOLD, score_threshold=SCORE_THRESHOLD)
image = utils.draw_bbox(original_image, predictions)
# image = utils.draw_bbox(image_data*255, predictions)
image = Image.fromarray(image.astype(np.uint8))
image.show()

detections = utils.count_detections(predictions)
print(detections)
print()
for key, value in detections.items():
    print(key, len([item for item in value if item]))
