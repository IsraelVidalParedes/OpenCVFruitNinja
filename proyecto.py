
import cv2
import numpy as np
import random

# Global variables
prev_frame = None
fruits = []

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

    # Compute the absolute difference between the current frame and previous frame
    diff = cv2.absdiff(prev_frame, frame)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 25, 255, cv2.THRESH_BINARY)
    return thresh

def main():
    global prev_frame, fruits

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
        motion_mask = detect_motion(frame, prev_frame)
        prev_frame = frame.copy()

        # Add fruits
        if random.random() < 0.02:
            fruits.append(Fruit(random.randint(0, frame.shape[1]), 0, random.randint(5, 10)))

        # Update and draw fruits
        for fruit in fruits:
            fruit.update()
            fruit.draw(frame)

            # Check for slicing
            if motion_mask is not None:
                if motion_mask[int(fruit.y), int(fruit.x)] == 255:
                    fruit.sliced = True

        # Remove sliced and out-of-bound fruits
        fruits = [fruit for fruit in fruits if not fruit.sliced and fruit.y < frame.shape[0]]

        cv2.imshow('Fruit Ninja', frame)

        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
