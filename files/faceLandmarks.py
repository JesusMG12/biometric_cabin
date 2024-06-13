#Last version 12102023
#REMINDER: https://github.com/google/mediapipe

import cv2
import mediapipe as mp
import numpy as np
import time
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh


# Obtain Specific Eye Landmark Indeces
# from MediaPipe Canonical Face Model 
# Source: https://github.com/google/mediapipe/blob/master/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png
left_eye_indices = [33, 160, 158, 133, 153, 144]
right_eye_indices = [362, 385, 387, 263, 373, 380]

# Obtain specific Mouth Landmark Indices
mouth_indices = [62, 73, 11, 303, 292, 404, 16, 180]

# Timing variables
start_time = None
eyes_closed_time = 0

# Frame Counter
frame_counter = 0

# Eye Aspect Ratio Threshold
ear_threshold = 0.85

# TEST BRANCH FOR MODIFYING EYE CLOSURE CALCULATION
# Function to calculate Eye Aspect Ratio
def calc_ear(eye_ldmks):
    #Obtain Eucledian distances between Eye Landmarks
    A = np.linalg.norm(np.array(eye_ldmks[1]) - np.array(eye_ldmks[5])) #P2 to P6
    B = np.linalg.norm(np.array(eye_ldmks[2]) - np.array(eye_ldmks[4])) #P3 to P5
    C = np.linalg.norm(np.array(eye_ldmks[0]) - np.array(eye_ldmks[3])) #P1 to P4
    ear = (abs(A) + abs(B)) / (2.0 * abs(C))
    return ear, A, B, C

# Webcam Input
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
#Initialize OpenCV Video Capture
#Camera id: Intel Realsense D435i = 4, Intel LiDAR RealSense L515 = 6.
camera_id = 6
cap = cv2.VideoCapture(camera_id)

# Establish font type
FONT = cv2.FONT_HERSHEY_SIMPLEX

with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as face_mesh:
  
  # Camera start time
  cam_time = time.time()

  # Camera loop start
  while cap.isOpened():
    frame_counter += 1
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(cv2.flip(image, 0), cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image)
    
    img_h, img_w, img_c = image.shape
    face_3d = []
    face_2d = []

    # Draw the face mesh annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_face_landmarks:
      for face_landmarks in results.multi_face_landmarks:
        # Draw face mesh in frame
        mp_drawing.draw_landmarks(
            image=image,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles
            .get_default_face_mesh_tesselation_style())
        # Draw face mesh contour around face, eyes, mouth and eyebrows in frame
        mp_drawing.draw_landmarks(
            image=image,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles
            .get_default_face_mesh_contours_style())
        # Draw boxes around irises in frame
        mp_drawing.draw_landmarks(
            image=image,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_IRISES,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles
            .get_default_face_mesh_iris_connections_style())
        
        ######################## HEAD POSE ESTIMATION ########################################
        
        for idx, lm in enumerate(face_landmarks.landmark):
            if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                if idx == 1:
                    nose_2d = (lm.x * img_w, lm.y * img_h)
                    nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)

                x, y = int(lm.x * img_w), int(lm.y * img_h)

                # Get the 2D Coordinates
                face_2d.append([x, y])

                # Get the 3D Coordinates
                face_3d.append([x, y, lm.z])

        # Convert it to the NumPy array
        face_2d = np.array(face_2d, dtype=np.float64)

        # Convert it to the NumPy array
        face_3d = np.array(face_3d, dtype=np.float64)

        # The camera matrix
        focal_length = 1 * img_w

        cam_matrix = np.array([ [focal_length, 0, img_h / 2],
                                [0, focal_length, img_w / 2],
                                [0, 0, 1]])

        # The distortion parameters
        dist_matrix = np.zeros((4, 1), dtype=np.float64)

        # Solve PnP
        success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

        # Get rotational matrix
        rmat, jac = cv2.Rodrigues(rot_vec)

        # Get angles
        angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

        # Get the y rotation degree
        x = angles[0] * 360
        y = angles[1] * 360
        z = angles[2] * 360
          

        # See where the user's head tilting
        if y < -10:
            text = "Looking Left"
        elif y > 10:
            text = "Looking Right"
        elif x < -10:
            text = "Looking Down"
        elif x > 10:
            text = "Looking Up"
        else:
            text = "Forward"

        # Display the nose direction
        nose_3d_projection, jacobian = cv2.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)

        p1 = (int(nose_2d[0]), int(nose_2d[1]))
        p2 = (int(nose_2d[0] + y * 10) , int(nose_2d[1] - x * 10))
            
        cv2.line(image, p1, p2, (255, 0, 0), 3)

        # Add the text on the image
        cv2.putText(image, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
        cv2.putText(image, "x: " + str(np.round(x,2)), (20, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(image, "y: " + str(np.round(y,2)), (20, 350), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(image, "z: " + str(np.round(z,2)), (20, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        
        ######################## Highlight Eye and Mouth Landmarks #######################################

        # Highlight specific landmarks for the right eye
        for idx in left_eye_indices:
            x = int(face_landmarks.landmark[idx].x * image.shape[1]) # Frame X coordinate
            y = int(face_landmarks.landmark[idx].y * image.shape[0]) # Frame Y coordinate
            cv2.circle(image, (x, y), 3, (0, 255, 0), -1)  # Green color for left eye

        # Highlight specific landmarks for the right eye
        for idx in right_eye_indices:
            x = int(face_landmarks.landmark[idx].x * image.shape[1]) # Frame X coordinate
            y = int(face_landmarks.landmark[idx].y * image.shape[0]) # Frame Y coordinate
            cv2.circle(image, (x, y), 3, (0, 0, 255), -1)  # Red color for right eye
        
        # Highlight specific landmarks for the mouth
        for idx in mouth_indices:
            x = int(face_landmarks.landmark[idx].x * image.shape[1]) # Frame X coordinate
            y = int(face_landmarks.landmark[idx].y * image.shape[0]) # Frame Y coordinate
            cv2.circle(image, (x, y), 3, (255, 0, 0), -1)  # Blue color for mouth

        #Obtain x,y coordinates of eye landmarks
        left_eye_coords = [(int(landmark.x * image.shape[1]), int(landmark.y * image.shape[0])) for i, landmark in enumerate(face_landmarks.landmark) if i in left_eye_indices]
        right_eye_coords = [(int(landmark.x * image.shape[1]), int(landmark.y * image.shape[0])) for i, landmark in enumerate(face_landmarks.landmark) if i in right_eye_indices]

        #print("Left Eye Coords:", left_eye_coords)

        ############# Calculate Eye Aspect Ratio (EAR) with eye coordinates ########################################

        #ear_left, A_left, B_left, C_left = calc_ear(left_eye_coords)
        #ear_right, A_right, B_right, C_right = calc_ear(right_eye_coords)

        #Average EAR
        #ear_avg = (ear_left + ear_right) / 2

        #TEST CHANGING THRESHOLD PARAMETERS
        #EAR Threshold for closed eyes
        #if ear_avg < ear_threshold:
        #   if start_time is None:
        #      start_time = time.time()
        #   else:
        #      eyes_closed_time = time.time() - start_time
        #   if eyes_closed_time > 3:
        #      cv2.putText(image, "ALERTA: OJOS CERRADOS!", (150, 400), FONT, 1, (0, 0, 255), 2)
        #else:
        #   start_time = None
        #   eyes_closed_time = 0
        
        # Display the left eye features
        #cv2.putText(image, f"Left Eye AR: {ear_left:.2f}", (10, 60), FONT, 1, (0, 0, 255), 2)
        #cv2.putText(image, f"Left Eye VDist1: {A_left:.2f}", (10, 180), FONT, 1, (0, 0, 255), 2)
        #cv2.putText(image, f"Left Eye VDist2: {B_left:.2f}", (10, 210), FONT, 1, (0, 0, 255), 2)
        #cv2.putText(image, f"Left Eye HDist: {C_left:.2f}", (10, 240), FONT, 1, (0, 0, 255), 2)
        
        # Display the right eye features
        #cv2.putText(image, f"Right Eye AR: {ear_right:.2f}", (10, 90), FONT, 1, (0, 255, 0), 2)
        #cv2.putText(image, f"Right Eye VDist1: {A_right:.2f}", (210, 270), FONT, 1, (0, 255, 0), 2)
        #cv2.putText(image, f"Right Eye VDist2: {B_right:.2f}", (210, 300), FONT, 1, (0, 255, 0), 2)
        #cv2.putText(image, f"Right Eye HDist: {C_right:.2f}", (210, 330), FONT, 1, (0, 255, 0), 2)

        # Display the average EAR
        #cv2.putText(image, f"Average Eye AR: {ear_avg:.2f}", (10, 120), FONT, 1, (0, 255, 255), 2)


        # Display the time eyes are closed
        #cv2.putText(image, f"Time eyes closed: {eyes_closed_time:.2f} s", (10, 30), FONT, 1, (0, 0, 0), 2)
              
    # Flip the image horizontally for a selfie-view display.
    #cv2.imshow('MediaPipe Face Mesh', cv2.flip(image, 1))

    cv2.imshow('MediaPipe Face Mesh', image)

    if cv2.waitKey(5) & 0xFF == 27: #Press Esc to end live camera feed
      break

cap.release()
cv2.destroyAllWindows()