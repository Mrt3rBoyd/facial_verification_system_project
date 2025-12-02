import cv2
from deepface import DeepFace
import os

known_folder = "known_face"

# detect to exist folder 
if not os.path.exists(known_folder):
    os.makedirs(known_folder)

# File handling function: log recognized filename and display name
def log_person(name):
    with open("recognized_log.txt", "a") as file:
        file.write(name + "\n")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    try:
        predictions = DeepFace.find(
            img_path=frame,
            db_path=known_folder,
            enforce_detection=False
        )

        if predictions and len(predictions[0]) > 0:
            identity = predictions[0].iloc[0]["identity"]
            name = os.path.splitext(os.path.basename(identity))[0]

            # Log file
            log_person(name)

            cv2.putText(frame, name, (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2)

        else:
            cv2.putText(frame, "Unknown", (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 0, 255), 2)

    except Exception as e:
        cv2.putText(frame, "No face", (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 255), 2)

    cv2.imshow("DeepFace Webcam", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
