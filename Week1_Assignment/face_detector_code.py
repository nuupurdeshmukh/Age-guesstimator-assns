import cv2

# reads image and converts it to numpy array (if fails, then returns None.)
img = cv2.imread("test.jpg")

# ADD: safety check (does not change logic)
if img is None:
    print("ERROR: test.jpg not found in the same folder")
    exit()

# convert the image to grayscale (face detection better hota hai in grayscale)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# loads the pre-trained Haar Cascade classifier for face detection
# ismei saare haar features loaded hai, reusing previous data of face detection
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# detects faces and returns [(x,y,w,h), ..]
# each tuple represents a face (x, y, width, height)
faces = face_cascade.detectMultiScale(gray, 1.3, 5)

print("Faces detected:", len(faces))

#  draws rectangles around detected faces
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

# displays the output image with boxed images and waits for the key press to close the window
# TRY OpenCV window first
cv2.imshow("Face Detection", img)
key = cv2.waitKey(0)
cv2.destroyAllWindows()

