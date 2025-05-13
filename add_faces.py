import cv2
import pickle
import numpy as np
import os

if not os.path.exists('data/'):
    os.makedirs('data/')

video = cv2.VideoCapture(0)
facedetect = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
faces_data = []

i = 0
aadhaar = input("Enter your Aadhaar number: ")
framesTotal = 51
captureAfterFrame = 2

while True:
    ret, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        crop_img = frame[y:y+h, x:x+w]
        resized_img = cv2.resize(crop_img, (50, 50))
        if len(faces_data) < framesTotal and i % captureAfterFrame == 0:
            faces_data.append(resized_img.flatten())
        i += 1
        cv2.putText(frame, str(len(faces_data)), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 255), 1)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 1)

    cv2.imshow('frame', frame)
    k = cv2.waitKey(1)
    if k == ord('q') or len(faces_data) >= framesTotal:
        break

video.release()
cv2.destroyAllWindows()

# Load existing database
if os.path.exists('data/database.pkl'):
    with open('data/database.pkl', 'rb') as f:
        database = pickle.load(f)
else:
    database = []

# Add new entries
for face in faces_data:
    database.append({"aadhaar": aadhaar, "face": face})

# Save back
with open('data/database.pkl', 'wb') as f:
    pickle.dump(database, f)

print("✅ Faces saved for Aadhaar:", aadhaar)
