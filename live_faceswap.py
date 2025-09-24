import cv2
import numpy as np

# Load swap face image
swap_image_path = r"D:\faceswap_live\swap_face.jpg"  # full path
swap_image = cv2.imread(swap_image_path)
if swap_image is None:
    print(f"Error: swap image not found at {swap_image_path}")
    exit()

# Load OpenCV face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Start webcam
cap = cv2.VideoCapture(0)
print("Press Q to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Resize swap face to match detected face size
        swap_resized = cv2.resize(swap_image, (w, h))

        # Create mask for seamless blending
        mask = 255 * np.ones(swap_resized.shape, swap_resized.dtype)
        center = (x + w // 2, y + h // 2)

        # Seamless clone (smooth overlay)
        frame = cv2.seamlessClone(swap_resized, frame, mask, center, cv2.NORMAL_CLONE)

    cv2.imshow("Live Face Swap", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()