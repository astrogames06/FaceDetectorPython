import cv2
from simplefacerec import SimpleFacerec


# encode faces from a folder
sfr = SimpleFacerec()
sfr.load_encoding_images("./images")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    # detect faces
    face_locations, face_names = sfr.detect_known_faces(frame)
    for face_pos, name in zip(face_locations, face_names):
        y1, x1, y2, x2 = face_pos[0], face_pos[1], face_pos[2], face_pos[3]

        cv2.putText(frame, name, (x1, y1-10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 4)

    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()