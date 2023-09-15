import cv2
import numpy as np
import random

# Global variables
prev_frame = None
fruits = []
bombs = []
score = 0
lives = 3

class Entity:
    def __init__(self, x, y, dy):
        self.x = x
        self.y = y
        self.dy = dy
        self.sliced = False

    def update(self):
        self.y += self.dy

class Fruit(Entity):
    def draw(self, frame):
        if not self.sliced:
            cv2.circle(frame, (self.x, self.y), 20, (0, 255, 0), -1)

class Bomb(Entity):
    def draw(self, frame):
        if not self.sliced:
            cv2.circle(frame, (self.x, self.y), 20, (0, 0, 255), -1)

def detect_motion(frame, prev_frame):
    if prev_frame is None:
        return None

    diff = cv2.absdiff(prev_frame, frame)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 25, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def main():
    global prev_frame, fruits, bombs, score, lives

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
        if random.random() < 0.01:  # Bombs are less frequent
            bombs.append(Bomb(random.randint(0, frame.shape[1]), 0, random.randint(5, 10)))

        entities = fruits + bombs
        for entity in entities:
            entity.update()
            entity.draw(frame)

            if contours:
                for contour in contours:
                    if cv2.pointPolygonTest(contour, (entity.x, entity.y), False) > 0:
                        entity.sliced = True
                        if isinstance(entity, Fruit):
                            score += 1
                        elif isinstance(entity, Bomb):
                            lives -= 1
                            if lives <= 0:
                                print("Game Over!")
                                cap.release()
                                cv2.destroyAllWindows()
                                return

        fruits = [fruit for fruit in fruits if not fruit.sliced and fruit.y < frame.shape[0]]
        bombs = [bomb for bomb in bombs if not bomb.sliced and bomb.y < frame.shape[0]]

        cv2.putText(frame, f"Score: {score}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame, f"Lives: {lives}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow('Fruit Ninja', frame)

        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
