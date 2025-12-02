import os
from deepface import DeepFace

## using OOP , filehandling as the attrbutrion
class FaceRecognitionSystem: ## 
    def __init__(self, known_faces_dir="known_face", log_file="recognized_log.txt"):
    
        self.known_faces_dir = known_faces_dir
        self.log_file = log_file
        self.already_logged = set()

        if not os.path.exists(self.known_faces_dir):
            os.makedirs(self.known_faces_dir)




    def recognize_face(self, frame, temp_img_path="temp_frame.jpg"):
        
        # Save frame temporarily
        from cv2 import imwrite
        imwrite(temp_img_path, frame)

        try:
            predictions = DeepFace.find(
                img_path=temp_img_path,
                db_path=self.known_faces_dir,
                enforce_detection=False
            )

            if predictions and len(predictions[0]) > 0:
                identity = predictions[0].iloc[0]["identity"]
                name = os.path.splitext(os.path.basename(identity))[0]
                self.log_person(name)
                return name
            else:
                return "Unknown"
        except Exception:
            return "No face detected"


