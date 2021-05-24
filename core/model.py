from typing import List
import numpy as np
import tensorflow as tf
from tensorflow.python.saved_model import tag_constants


class YOLOModel:
    """
    Class to make predictions
    with TensorfLow YOLO model
    """
    def __init__(self, weights_path: str):
        self.weights_path = weights_path

    def detect(self, 
                input_data: np.array,
                iou_threshold: float,
                score_threshold: float,
                max_output_size_per_class: int = 50,
                max_total_size: int = 50
                ) -> List:
        """Make predictions on image

        Args:
            input_data (np.array): Image array
            iou_threshold (float): IoU threshold for predictions
            score_threshold (float): Score threshold for predictions
            max_output_size_per_class (int): Max output size per class. Defaults to 50.
            max_total_size (int, optional): Max total size of detections. Defaults to 50.

        Returns:
            List: List with predictions
        """
        saved_model_loaded = tf.saved_model.load(self.weights_path, tags=[tag_constants.SERVING])
        infer = saved_model_loaded.signatures['serving_default']

        batch_data = tf.constant(input_data)
        pred_bbox = infer(batch_data)

        for _, value in pred_bbox.items():
            boxes = value[:, :, 0:4]
            pred_conf = value[:, :, 4:]

        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(
                pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])
                ),
            max_output_size_per_class=max_output_size_per_class,
            max_total_size=max_total_size,
            iou_threshold=iou_threshold,
            score_threshold=score_threshold
        )
        return [boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy()]
