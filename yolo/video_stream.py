import cv2

class VideoStream:
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src)

    def start(self):
        if not self.cap.isOpened():
            print("Error: Could not open video source.")
            return

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            yield frame

    def release(self):
        self.cap.release()
        cv2.destroyAllWindows()