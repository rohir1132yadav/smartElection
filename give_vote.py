from sklearn.neighbors import KNeighborsClassifier
import cv2
import pickle
import numpy as np
import os
import csv
import time
from datetime import datetime
from win32com.client import Dispatch

def speak(str1):
    speak = Dispatch("SAPI.SpVoice")
    speak.Speak(str1)

# Load database
with open('data/database.pkl', 'rb') as f:
    database = pickle.load(f)

FACES = np.array([entry['face'] for entry in database])
LABELS = [entry['aadhaar'] for entry in database]

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(FACES, LABELS)

video = cv2.VideoCapture(0)
facedetect = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

COL_NAMES = ['AADHAAR', 'VOTE', 'DATE', 'TIME']

detected_aadhaar = None

# Load and resize party logos
logo_size = (200, 100)

def load_logo(path):
    logo = cv2.imread(path)
    if logo is None:
        print(f"Warning: Failed to load logo at {path}")
        return None
    return cv2.resize(logo, logo_size)

bjp_logo = load_logo("data/bjp_logo.png")
congress_logo = load_logo("data/congress_logo.png")
aap_logo = load_logo("data/aap_logo.png")
nota_logo = load_logo("data/nota_logo.png")

def draw_buttons(frame, buttons):
    frame_height, frame_width = frame.shape[:2]

    for name, (x1, y1, x2, y2, logo) in buttons.items():
        if logo is None:
            print(f"Warning: Logo for {name} is missing.")
            continue

        width = x2 - x1
        height = y2 - y1

        # Ensure the region is within the frame
        if x2 > frame_width or y2 > frame_height or x1 < 0 or y1 < 0:
            print(f"Warning: Region for {name} button exceeds frame size.")
            continue

        resized_logo = cv2.resize(logo, (width, height))
        frame[y1:y2, x1:x2] = resized_logo

def check_if_exists(value):
    try:
        with open("vote.csv", "r") as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                if row and row[0] == value:
                    return True
    except FileNotFoundError:
        pass
    return False

def record_vote(aadhaar, party_name):
    ts = time.time()
    date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
    timestamp = datetime.fromtimestamp(ts).strftime("%H-%M-%S")
    exist = os.path.isfile("vote.csv")

    with open("vote.csv", "a", newline='') as csvfile:
        writer = csv.writer(csvfile)
        if not exist:
            writer.writerow(COL_NAMES)
        attendance = [aadhaar, party_name, date, timestamp]
        writer.writerow(attendance)

    speak("YOUR VOTE HAS BEEN RECORDED. THANK YOU.")
    time.sleep(2)

def click_vote(x, y, buttons):
    for party, (x1, y1, x2, y2, _) in buttons.items():
        if x1 <= x <= x2 and y1 <= y <= y2:
            return party
    return None

# Mouse click event
def mouse_callback(event, x, y, flags, param):
    global detected_aadhaar
    buttons = param  # Buttons are passed via param

    if event == cv2.EVENT_LBUTTONDOWN and detected_aadhaar:
        party = click_vote(x, y, buttons)
        if party:
            if check_if_exists(detected_aadhaar):
                speak("YOU HAVE ALREADY VOTED")
            else:
                record_vote(detected_aadhaar, party)
            detected_aadhaar = None  # Reset after voting

cv2.namedWindow('Voting Booth')

while True:
    ret, frame = video.read()
    if not ret:
        print("Failed to capture frame.")
        continue

    frame_height, frame_width = frame.shape[:2]

    buttons = {
        "BJP": (10, frame_height - 110, 210, frame_height - 10, bjp_logo),
        "CONGRESS": (220, frame_height - 110, 420, frame_height - 10, congress_logo),
        "AAP": (430, frame_height - 110, 630, frame_height - 10, aap_logo),
        "NOTA": (640, frame_height - 110, 840, frame_height - 10, nota_logo)
    }
    if buttons["NOTA"][2] > frame_width:
         buttons["NOTA"] = (frame_width - 200, frame_height - 110, frame_width, frame_height - 10, nota_logo)

    cv2.setMouseCallback('Voting Booth', mouse_callback, param=buttons)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)

    if len(faces) > 0 and detected_aadhaar is None:
        for (x, y, w, h) in faces:
            crop_img = frame[y:y+h, x:x+w]
            resized_img = cv2.resize(crop_img, (50, 50)).flatten().reshape(1, -1)
            output = knn.predict(resized_img)[0]
            detected_aadhaar = output
            print("Detected Aadhaar:", detected_aadhaar)
            speak(f"Aadhaar detected {detected_aadhaar}")
            break

    if detected_aadhaar:
        cv2.putText(frame, f"Aadhaar: {detected_aadhaar}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)
        draw_buttons(frame, buttons)

    cv2.imshow('Voting Booth', frame)
    if cv2.waitKey(1) == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
