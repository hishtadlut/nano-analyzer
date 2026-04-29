import cv2


def verify_face(frame, enrolled_template):
    detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    faces = detector.detectMultiScale(frame)
    return len(faces) > 0 and enrolled_template is not None
