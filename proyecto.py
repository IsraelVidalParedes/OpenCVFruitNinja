import cv2
import numpy as np
import random

# Global variables
prev_frame = None
fruits = []
score = 0

class Fruit:
    def __init__(self, x, y, dy):
        self.x = x
        self.y = y
        self.dy = dy
        self.sliced = False

    def update(self):
        self.y += self.dy

    def draw(self, frame):
        if not self.sliced:
            cv2.circle(frame, (self.x, self.y), 20, (0, 255, 0), -1)

def detect_motion(frame, prev_frame):
    if prev_frame is None:
        return None

    diff = cv2.absdiff(prev_frame, frame)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 25, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def main():
    global prev_frame, fruits, score

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Can't receive frame. Exiting ...")
            break

        frame = cv2.flip(frame, 1)
        contours = detect_motion(frame, prev_frame)
        prev_frame = frame.copy()

        if random.random() < 0.02:
            fruits.append(Fruit(random.randint(0, frame.shape[1]), 0, random.randint(5, 10)))

        for fruit in fruits:
            fruit.update()
            fruit.draw(frame)

            if contours:
                for contour in contours:
                    if cv2.pointPolygonTest(contour, (fruit.x, fruit.y), False) > 0:
                        fruit.sliced = True
                        score += 1

        fruits = [fruit for fruit in fruits if not fruit.sliced and fruit.y < frame.shape[0]]

        cv2.putText(frame, f"Score: {score}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow('Fruit Ninja', frame)

        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
