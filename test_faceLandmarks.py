#!/usr/bin/env python3

import os

desired_env = "mp_env"

# Check if the environment is active
def is_env_active(env_name):
    try:
        return env_name == os.environ['VIRTUAL_ENV'].split('/')[-1]
    except KeyError:
        return False

# Activate the environment if it's not active
if not is_env_active(desired_env):
    activate_script = f"source /home/dev/biometric_cabin/{desired_env}/bin/activate"
    os.system(activate_script)



import cv2
import mediapipe as mp
import numpy as np
import time

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh


left_eye_indices = [33, 160, 158, 133, 153, 144]
right_eye_indices = [362, 385, 387, 263, 373, 380]

#Timing variables
start_time = None
eyes_closed_time = 0


#Eye Aspect Ratio Threshold
ear_threshold = 0.85


#Function to calculate Eye Aspect Ratio
def calc_ear(eye_ldmks):
    #Obtain Eucledian distances between Eye Landmarks
    A = np.linalg.norm(np.array(eye_ldmks[1]) - np.array(eye_ldmks[5])) #P2 to P6
    B = np.linalg.norm(np.array(eye_ldmks[2]) - np.array(eye_ldmks[4])) #P3 to P5
    C = np.linalg.norm(np.array(eye_ldmks[0]) - np.array(eye_ldmks[3])) #P1 to P4
    ear = (A + B) / (2.0 * C)
    return ear

# Webcam Input
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
#Initialize OpenCV Video Capture
cap = cv2.VideoCapture(6) #Camera Index (For Intel RealSense use 4)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)

with mp_face_mesh.FaceMesh(
    max_num_faces=6,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as face_mesh:
  
  print("Press Esc on the Face Mesh Window, or Close Terminal to Exit")
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(cv2.flip(image, -1), cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image)
    
    # Draw the face mesh annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_face_landmarks:
      for face_landmarks in results.multi_face_landmarks:
        mp_drawing.draw_landmarks(
            image=image,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles
            .get_default_face_mesh_tesselation_style())
        mp_drawing.draw_landmarks(
            image=image,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles
            .get_default_face_mesh_contours_style())
        mp_drawing.draw_landmarks(
            image=image,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_IRISES,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles
            .get_default_face_mesh_iris_connections_style())
        
        # Highlight specific landmarks for the right eye
        for idx in left_eye_indices:
            x = int(face_landmarks.landmark[idx].x * image.shape[1])
            y = int(face_landmarks.landmark[idx].y * image.shape[0])
            cv2.circle(image, (x, y), 3, (0, 255, 0), -1)  # Green color for left eye

        # Highlight specific landmarks for the right eye
        for idx in right_eye_indices:
            x = int(face_landmarks.landmark[idx].x * image.shape[1])
            y = int(face_landmarks.landmark[idx].y * image.shape[0])
            cv2.circle(image, (x, y), 3, (0, 0, 255), -1)  # Red color for right eye

        #Obtain x,y coordinates of eye landmarks
        left_eye_coords = [(int(landmark.x * image.shape[1]), int(landmark.y * image.shape[0])) for i, landmark in enumerate(face_landmarks.landmark) if i in left_eye_indices]
        right_eye_coords = [(int(landmark.x * image.shape[1]), int(landmark.y * image.shape[0])) for i, landmark in enumerate(face_landmarks.landmark) if i in right_eye_indices]

        #Calculate Eye Aspect Ratio (EAR) with eye coordinates
        ear_left = calc_ear(left_eye_coords)
        ear_right = calc_ear(right_eye_coords)

        #Average EAR
        ear_avg = (ear_left + ear_right) / 2


        #EAR Threshold for closed eyes
        if ear_avg < ear_threshold:
           if start_time is None:
              start_time = time.time()
           else:
              eyes_closed_time = time.time() - start_time
           if eyes_closed_time > 3:
              temp = 0
              #cv2.putText(image, "ALERTA: OJOS CERRADOS!", (150, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
           start_time = None
           eyes_closed_time = 0
        
        # Display the left EAR
        #cv2.putText(image, f"Left Eye AR: {ear_left:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Display the right EAR
        #cv2.putText(image, f"Right Eye AR: {ear_right:.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display the average EAR
        #cv2.putText(image, f"Average Eye AR: {ear_avg:.2f}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)


        # Display the time eyes are closed
        #cv2.putText(image, f"Time eyes closed: {eyes_closed_time:.2f} s", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
              
    # Flip the image horizontally for a selfie-view display.
    #cv2.imshow('MediaPipe Face Mesh', cv2.flip(image, 1))

    cv2.imshow('MediaPipe Face Mesh - Press Esc or Close Terminal to Exit', image)

    if cv2.waitKey(5) & 0xFF == 27: #Press Esc to end live camera feed
      break

cap.release()
cv2.destroyAllWindows()