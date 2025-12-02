# main.py
import cv2
from deepface_cam import FaceRecognitionSystem


### Filhandling 
def main():
    fr_system = FaceRecognitionSystem(known_faces_dir="known_face", log_file="recognized_log.txt")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[ERROR] Cannot access webcam.")
        return

    print("Press 'q' to quit.")

    frame_count = 0
    PROCESS_EVERY = 5  

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % PROCESS_EVERY == 0:
            name = fr_system.recognize_face(frame)
        else:
            name = "Processing..."

        cv2.putText(frame, f"{name}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0) if name != "Unknown" else (0, 0, 255), 2)
        cv2.putText(frame, "Press 'q' to quit", (10, frame.shape[0]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

        cv2.imshow("DeepFace Webcam", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Webcam closed.")

if __name__ == "__main__":
    main()



