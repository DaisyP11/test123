import cv2
import traceback
import numpy as np
from numpy.typing import ArrayLike
import matplotlib.pyplot as plt

face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml" # type: ignore
)

def extractPersonImg(img: np.ndarray) -> np.ndarray|ArrayLike|None:
    try:
        if img is None:
            raise ValueError("Input image is None.")

        height, width, _ = img.shape
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        face = face_classifier.detectMultiScale(
            gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
        )

        if len(face) == 0:
            raise ValueError("No faces detected in the image.")

        x, y, w, h = face[0]
        x = int(max(0, x - 0.5 * w))
        y = int(max(0, y - 0.5 * h))
        w = int(w + 1 * w)
        h = int(h + 1.5 * h)

        cropped_face = img[y:min(y + h, height), x:min(x + w, width)]
        img_rgb = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB)

        return img_rgb

    except Exception as e:
        traceback.print_exc()
        print("Error:", e)
        return None

if __name__ == "__main__":
    imagePath = "/run/media/spritan/38c3181a-2d49-4ccb-bdbe-e934afa1eedc/handwritten_text_detection_and_recognition/dataInput/page/Screenshot_20240417_204206.png"
    img = cv2.imread(imagePath)
    img_rgb = extractPersonImg(img)
    plt.figure(figsize=(20,10))
    plt.imshow(img_rgb) # type: ignore
    plt.axis('off')
    plt.show()