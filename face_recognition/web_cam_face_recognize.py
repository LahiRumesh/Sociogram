import face_recognition
import cv2
import os
import numpy as np

Faces_Folder="faces"

person_name,face_img,face_names,face_locations,face_encodings=([] for i in range(5))
process_this_frame = True
video_capture = cv2.VideoCapture(0)

for name in os.listdir(Faces_Folder):
    for img_name in os.listdir(f'{Faces_Folder}/{name}'):
        load_img=face_recognition.load_image_file(f'{Faces_Folder}/{name}/{img_name}')
        face_recog=face_recognition.face_encodings(load_img)[0]
        face_img.append(face_recog)
        person_name.append(img_name.split(".")[0])

while True:
    ret, frame = video_capture.read()
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = small_frame[:, :, ::-1]
    if process_this_frame:
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(face_img, face_encoding)
            name = "Unknown"
            face_distances = face_recognition.face_distance(face_img, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = person_name[best_match_index]

            face_names.append(name)

    process_this_frame = not process_this_frame

    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 1)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    cv2.imshow('Output', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()



