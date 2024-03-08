import cv2
import os
from models import EyesModel
import torch
from pygame import mixer
import numpy as np
from torchvision import transforms
mixer.init()
sound = mixer.Sound('alarm.wav')

face = cv2.CascadeClassifier('haar cascade files\haarcascade_frontalface_alt.xml')
leye = cv2.CascadeClassifier('haar cascade files\haarcascade_lefteye_2splits.xml')
reye = cv2.CascadeClassifier('haar cascade files\haarcascade_righteye_2splits.xml')

lbl = ['Close', 'Open']

model = EyesModel()
model.load_state_dict(torch.load('models/eyes_model.pth'))
model.eval()
path = "./logs"
cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
count = 0
score = 0
thicc = 2
rpred = [99]
lpred = [99]

while True:
    ret, frame = cap.read()
    height, width = frame.shape[:2]
    faces = face.detectMultiScale(frame, minNeighbors=5, scaleFactor=1.1, minSize=(25, 25))
    left_eye = leye.detectMultiScale(frame)
    right_eye = reye.detectMultiScale(frame)

    cv2.rectangle(frame, (0, height - 50), (200, height), (0, 0, 0), thickness=cv2.FILLED)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (100, 100, 100), 1)

    for (x, y, w, h) in right_eye:
        r_eye = frame[y:y + h, x:x + w]
        count = count + 1

        r_eye = cv2.resize(r_eye, (64, 64))

        r_eye = r_eye.astype('float32')
        r_eye = r_eye.reshape(-1, 64, 64)
        r_eye = np.expand_dims(r_eye, axis=0)
        r_eye = torch.from_numpy(r_eye)
        with torch.no_grad():
            rpred = np.argmax(model(r_eye))
        if rpred == 1:
            lbl = 'Open'
        else:
            lbl = 'Closed'

    for (x, y, w, h) in left_eye:
        l_eye = frame[y:y + h, x:x + w]
        count = count + 1

        l_eye = cv2.resize(l_eye, (64, 64))
        l_eye = l_eye.astype('float32')
        l_eye = l_eye.reshape(-1, 64, 64)
        l_eye = np.expand_dims(l_eye, axis=0)
        l_eye = torch.from_numpy(l_eye)
        with torch.no_grad():
            lpred = np.argmax(model(l_eye))

        if lpred == 1:
            lbl = 'Open'
        else:
            lbl = 'Closed'
        break

    if rpred == 0 and lpred == 0:
        score = score + 1
        cv2.putText(frame, "Closed", (10, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
    # if(rpred[0]==1 or lpred[0]==1):
    else:
        score = score - 1
        cv2.putText(frame, "Open", (10, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

    if score < 0:
        score = 0
    cv2.putText(frame, 'Score:' + str(score), (100, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
    if score > 15:
        # person is feeling sleepy so we beep the alarm
        cv2.imwrite(os.path.join(path, 'image.jpg'), frame)
        try:
            sound.play()

        except:  # isplaying = False
            pass
        if thicc < 16:
            thicc = thicc + 2
        else:
            thicc = thicc - 2
            if thicc < 2:
                thicc = 2
        cv2.rectangle(frame, (0, 0), (width, height), (0, 0, 255), thicc)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

#%%
