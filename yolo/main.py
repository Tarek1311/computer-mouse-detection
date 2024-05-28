import os
from dotenv import load_dotenv
import cv2
from detection.py import ObjectDetector
from video_stream.py import VideoStream

def main():
    load_dotenv()
    api_key = os.getenv('API_KEY')

    yolo_cfg = 'yolo/yolov3.cfg'
    yolo_weights = 'yolo/yolov3.weights'
    yolo_names = 'yolo/coco.names'

    detector = ObjectDetector(yolo_cfg, yolo_weights, yolo_names)

    stream = VideoStream()
    for frame in stream.start():
        detections = detector.detect(frame)
        frame = detector.annotate_frame(frame, detections)
        cv2.imshow('YOLO Object Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    stream.release()

if __name__ == '__main__':
    main()
