
import cv2
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
print("camera OK" if ret else "camera not found (try index 1)")
cap.release()
