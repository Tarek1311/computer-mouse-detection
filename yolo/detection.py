import cv2
import numpy as np
import torch

class ObjectDetector:
    def __init__(self, model_cfg, model_weights, class_names, conf_threshold=0.5, nms_threshold=0.4):
        self.model = cv2.dnn.readNetFromDarknet(model_cfg, model_weights)
        self.model.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.model.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold

        with open(class_names, 'r') as f:
            self.classes = [line.strip() for line in f.readlines()]

    def detect(self, frame):
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
        self.model.setInput(blob)
        layer_names = self.model.getLayerNames()
        output_layers = [layer_names[i[0] - 1] for i in self.model.getUnconnectedOutLayers()]
        detections = self.model.forward(output_layers)

        boxes = []
        confidences = []
        class_ids = []

        for output in detections:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > self.conf_threshold:
                    box = detection[0:4] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
                    (centerX, centerY, width, height) = box.astype("int")
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        idxs = cv2.dnn.NMSBoxes(boxes, confidences, self.conf_threshold, self.nms_threshold)
        results = []
        if len(idxs) > 0:
            for i in idxs.flatten():
                x, y, w, h = boxes[i]
                results.append((x, y, w, h, self.classes[class_ids[i]], confidences[i]))

        return results

    def annotate_frame(self, frame, detections):
        for (x, y, w, h, label, confidence) in detections:
            color = (0, 255, 0)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            text = f"{label}: {confidence:.2f}"
            cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return frame
