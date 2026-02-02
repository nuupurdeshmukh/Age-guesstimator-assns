import cv2
import numpy as np

# Face detector
FACE_PROTO = "models/deploy.prototxt"
FACE_MODEL = "models/res10_300x300_ssd_iter_140000_fp16.caffemodel"

# Age model
AGE_PROTO = "models/age_deploy.prototxt"
AGE_MODEL = "models/age_net.caffemodel"

# Gender model
GENDER_PROTO = "models/gender_deploy.prototxt"
GENDER_MODEL = "models/gender_net.caffemodel"

AGE_BUCKETS = ['(0-2)', '(4-6)', '(8-12)', '(15-20)',
               '(25-32)', '(38-43)', '(48-53)', '(60-100)']

GENDER_LIST = ['Male', 'Female']

# Load networks
faceNet = cv2.dnn.readNet(FACE_MODEL, FACE_PROTO)
ageNet = cv2.dnn.readNet(AGE_MODEL, AGE_PROTO)
genderNet = cv2.dnn.readNet(GENDER_MODEL, GENDER_PROTO)

# Load image (test.jpeg in same folder)
img = cv2.imread("test.jpeg")

if img is None:
    print("Image not found!")
    input("Press Enter...")
    exit()

h, w = img.shape[:2]

# ---- FACE DETECTION (improved) ----
blob = cv2.dnn.blobFromImage(
    img, 1.0, (400, 400),   # bigger input helps mid-distance faces
    [104, 117, 123], True, False
)

faceNet.setInput(blob)
detections = faceNet.forward()

faces = []

for i in range(detections.shape[2]):
    conf = detections[0, 0, i, 2]
    if conf > 0.3:   # lowered threshold
        x1 = int(detections[0, 0, i, 3] * w)
        y1 = int(detections[0, 0, i, 4] * h)
        x2 = int(detections[0, 0, i, 5] * w)
        y2 = int(detections[0, 0, i, 6] * h)
        faces.append([x1, y1, x2, y2])

if len(faces) == 0:
    print("No face detected.")
    input("Press Enter...")
    exit()

# ---- AGE + GENDER FOR EACH FACE ----
for box in faces:

    face = img[max(0, box[1]-20):min(box[3]+20, h-1),
               max(0, box[0]-20):min(box[2]+20, w-1)]

    blob = cv2.dnn.blobFromImage(
        face, 1.0, (227, 227),
        (78.4263377603, 87.7689143744, 114.895847746),
        swapRB=False
    )

    genderNet.setInput(blob)
    gender = GENDER_LIST[genderNet.forward()[0].argmax()]

    ageNet.setInput(blob)
    age = AGE_BUCKETS[ageNet.forward()[0].argmax()]

    label = f"{gender}, {age}"

    cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
    cv2.putText(img, label, (box[0], box[1]-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

# Show result
cv2.imshow("Age-Gender", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

input("Press Enter to close...")
