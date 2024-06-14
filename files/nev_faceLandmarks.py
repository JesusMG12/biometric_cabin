#Sources: https://github.com/google/mediapipe

import cv2
import mediapipe as mp
import numpy as np
import time
import csv
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

# Select Mode: 0 for Face Features, 1 for image and recording
mode = 0
def change_mode(val):
   global mode
   mode = val

# Obtain Specific Eye Landmark Indeces
# from MediaPipe Canonical Face Model 
# Source: https://github.com/google/mediapipe/blob/master/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png
left_eye_indices = [33, 160, 158, 133, 153, 144]
right_eye_indices = [362, 385, 387, 263, 373, 380]

# Obtain specific Mouth Landmark Indices
mouth_indices = [78, 81, 13, 311, 308, 402, 14, 178]

# Timing variables EYES
start_time = None
eyes_closed_time = 0

# Timing variables MOUTH
start_time_mouth = None
mouth_opened_time = 0

prev_frame_time = 0
new_frame_time = 0
elapsed_time = 0
fps = 0

# Frame Counter
frame_counter = 0

# Initialize face features
face_detected = 0
ear_left = 0
ear_right = 0
ear_avg = 0
mar = 0
x_rot = 0
y_rot = 0
z_rot = 0

# THRESHOLDS


# Eye Aspect Ratio Threshold
ear_threshold = 0.15

# Mouth Aspect Ratio Threshold
mar_threshold = 0.5



# Function to calculate Eye Aspect Ratio
def calc_ear(eye_ldmks):
    #Obtain Eucledian distances between Eye Landmarks
    A = np.linalg.norm(np.array(eye_ldmks[1]) - np.array(eye_ldmks[5])) #P2 to P6
    B = np.linalg.norm(np.array(eye_ldmks[2]) - np.array(eye_ldmks[4])) #P3 to P5
    C = np.linalg.norm(np.array(eye_ldmks[0]) - np.array(eye_ldmks[3])) #P1 to P4
    ear = (abs(A) + abs(B)) / (2.0 * abs(C))
    return ear, A, B, C

# Function to calculate Mouth Aspect Ratio
# mouth_indices = [P1, P2, P3, P4, P5, P6, P7, P8]
# mouth_indices = [78, 81, 13, 311, 308, 402, 14, 178]
def calc_mar(mouth_ldmks):
   # Obtain Eucledian distances between Mouth Landmarks
   A = np.linalg.norm(np.array(mouth_ldmks[1]) - np.array(mouth_ldmks[7])) #P2 to P8
   B = np.linalg.norm(np.array(mouth_ldmks[2]) - np.array(mouth_ldmks[6])) #P3 to P7
   C = np.linalg.norm(np.array(mouth_ldmks[3]) - np.array(mouth_ldmks[5])) #P4 to P6
   D = np.linalg.norm(np.array(mouth_ldmks[0]) - np.array(mouth_ldmks[4])) #P1 to P5
   mar = (abs(A) + abs(B) + abs(C)) / (2.0 * abs(D))
   return mar, A, B, C, D

# Webcam Input
videoname = 'test.avi'
# videoname = input('Enter video filename: ') + '.avi'
print(f"Output file set to {videoname}")

drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
#Initialize OpenCV Video Capture
cap = cv2.VideoCapture(6)

# Set the width and height of the frame
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960) #960
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540) #540

# Create named window for camera feed
cv2.namedWindow('Facial Features')

# Add trackbar for mode toggling
cv2.createTrackbar('Mode', 'Facial Features', 0, 1, change_mode)

# Initialize video writer (set the desired output file name and format)
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out = cv2.VideoWriter(videoname, fourcc, 16.61, (960, 540))

# Initialize recording flag
recording = False

# Establish font type and size
FONT = cv2.FONT_HERSHEY_COMPLEX
ldmk_font_size = 0.3
info_font_size = 0.5

# Establish text background color
text_background = (0, 0, 0)

# Establish Landmark Information Display Location
info_left_eye_x = 10
text_x_offset = 240
info_ldmk_y = 80
info_right_eye_x = info_left_eye_x + text_x_offset
info_ear_avg = int((info_left_eye_x + info_right_eye_x)/2)
info_mouth_x = info_right_eye_x + text_x_offset

# Initialize lists to store data
timestamps = []
face_detections = []
ear_left_val = []
ear_right_val = []
ear_avg_val = []
mar_val = []
head_rot_x = []
head_rot_y = []
head_rot_z = []


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
   #  # Measure video framerate
   #  new_frame_time = time.time()
   #  fps = 1/(new_frame_time-prev_frame_time)
   #  prev_frame_time = new_frame_time
   #  print(fps)
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    if cv2.waitKey(1) & 0xFF == ord('m'):
       mode = 0 if mode == 1 else 1

    

    if mode == 0:
       #print('Mode 1: Facial Features')

       # To improve performance, optionally mark the image as not writeable to
       # pass by reference.
       image.flags.writeable = False
       image = cv2.cvtColor(cv2.flip(image, 0), cv2.COLOR_BGR2RGB)
       # Display Face Features Mode
       cv2.putText(image, "Facial Features", (10, 30), FONT, 1, (255, 255, 255), 2)
       results = face_mesh.process(image)

       if results.multi_face_landmarks:
          face_detected = 1
       else:
          face_detected = 0
          ear_left = 0
          ear_right = 0
          ear_avg = 0
          mar = 0
          x_rot = 0
          y_rot = 0
          z_rot = 0
    
       img_h, img_w, img_c = image.shape
       face_3d = []
       face_2d = []
       # print(img_w, img_h, img_c)
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
         #   mp_drawing.draw_landmarks(
         #      image=image,
         #      landmark_list=face_landmarks,
         #      connections=mp_face_mesh.FACEMESH_CONTOURS,
         #      landmark_drawing_spec=None,
         #      connection_drawing_spec=mp_drawing_styles
         #      .get_default_face_mesh_contours_style())
           # Draw boxes around irises in frame
           """ mp_drawing.draw_landmarks(
              image=image,
              landmark_list=face_landmarks,
              connections=mp_face_mesh.FACEMESH_IRISES,
              landmark_drawing_spec=None,
              connection_drawing_spec=mp_drawing_styles
              .get_default_face_mesh_iris_connections_style()) """
           
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
           x_rot = angles[0] * 360 * 5
           y_rot = angles[1] * 360 * 5
           z_rot = angles[2] * 360
          

           # See where the user's head tilting
           if y_rot < -10:
              text = "Looking Left"
           elif y_rot > 10:
              text = "Looking Right"
           elif x_rot < -10:
              text = "Looking Down"
           elif x_rot > 10:
              text = "Looking Up"
           else:
              text = "Forward"

           # Display the nose direction
           nose_3d_projection, jacobian = cv2.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)

           p1 = (int(nose_2d[0]), int(nose_2d[1]))
           p2 = (int(nose_2d[0] + y_rot * 10) , int(nose_2d[1] - x_rot * 10))
            
           cv2.arrowedLine(image, p1, p2, (255, 255, 255), 3)

           # Add the text on the image
           #First draw background rectangle (For consistent contrast)
           cv2.rectangle(image, (info_left_eye_x, info_ldmk_y-20), (info_left_eye_x+135, info_ldmk_y+50), text_background, -1)

           #cv2.putText(image, text, (int(img_w/2)-150, 50), FONT, 1.5, (0, 255, 0), 2)
           text_y_offset = 0
           cv2.putText(image, f"HEAD ROTATION", (info_left_eye_x, info_ldmk_y+text_y_offset), FONT, info_font_size, (255, 255, 255), 2)
           text_y_offset += 20
           cv2.putText(image, f"x: {str(np.round(x_rot,2))} deg", (info_left_eye_x, info_ldmk_y+text_y_offset), FONT, info_font_size, (128, 128, 128), 2)
           text_y_offset += 15
           cv2.putText(image, f"y: {str(np.round(y_rot,2))} deg", (info_left_eye_x, info_ldmk_y+text_y_offset), FONT, info_font_size, (128, 128, 128), 2)
           text_y_offset += 15
           cv2.putText(image, f"z: {str(np.round(z_rot,2))} deg", (info_left_eye_x, info_ldmk_y+text_y_offset), FONT, info_font_size, (128, 128, 128), 2)
           text_y_offset += 25
        
        
           ######################## Highlight & Store Eye and Mouth Landmarks #######################################

           # Highlight specific landmarks for the left eye
           left_eye_coords = []
           # Draw text background rectangle
           cv2.rectangle(image, (info_left_eye_x, info_ldmk_y+text_y_offset-20), (info_left_eye_x+135, info_ldmk_y+text_y_offset+225), text_background, -1)
           cv2.putText(image, f"EYE FEATs", (info_left_eye_x, info_ldmk_y+text_y_offset), FONT, info_font_size, (255, 255, 255), 2)
           text_y_offset += 20
           cv2.putText(image, f"Left Eye LMKs:", (info_left_eye_x, info_ldmk_y+text_y_offset), FONT, info_font_size, (0, 255, 0), 2)
           for i, idx in enumerate(left_eye_indices):
            # Obtain x,y coordinates of left eye landmarks
            x = int(face_landmarks.landmark[idx].x * image.shape[1]) # Frame X coordinate
            y = int(face_landmarks.landmark[idx].y * image.shape[0]) # Frame Y coordinate

            # Store landmark coordinates in list
            left_eye_coords.append((x, y))

            cv2.circle(image, (x, y), 3, (0, 255, 0), -1)  # Green color for left eye
            cv2.putText(image, f"P{i+1}", (x, y), FONT, ldmk_font_size, (0, 255, 0), 1)

            # Show landmark coordinates
            #cv2.putText(image, f"P{i+1}: ({x}, {y})", (info_left_eye_x, info_ldmk_y+text_y_offset), FONT, info_font_size, (0, 255, 0), 2)
            #text_y_offset += 15


           ear_left, A_left, B_left, C_left = calc_ear(left_eye_coords)

           # Display the left eye features
           text_y_offset += 15
           cv2.putText(image, f"EAR: {ear_left:.2f}", (info_left_eye_x, info_ldmk_y+text_y_offset), FONT, info_font_size, (0, 255, 0), 2)
           text_y_offset += 15
           cv2.putText(image, f"P2-P6: {A_left:.2f}", (info_left_eye_x, info_ldmk_y+text_y_offset), FONT, info_font_size, (0, 255, 0), 2)
           text_y_offset += 15
           cv2.putText(image, f"P3-P5: {B_left:.2f}", (info_left_eye_x, info_ldmk_y+text_y_offset), FONT, info_font_size, (0, 255, 0), 2)
           text_y_offset += 15
           cv2.putText(image, f"P1-P4: {C_left:.2f}", (info_left_eye_x, info_ldmk_y+text_y_offset), FONT, info_font_size, (0, 255, 0), 2)


           ### RIGHT EYE ###
           # Highlight specific landmarks for the right eye
           right_eye_coords = []
           text_y_offset += 30
           cv2.putText(image, f"Right Eye LMKs:", (info_left_eye_x, info_ldmk_y+text_y_offset), FONT, info_font_size, (0, 0, 255), 2)
           for i, idx in enumerate(right_eye_indices):
            # Obtain x,y coordinates of right eye landmarks
            x = int(face_landmarks.landmark[idx].x * image.shape[1]) # Frame X coordinate
            y = int(face_landmarks.landmark[idx].y * image.shape[0]) # Frame Y coordinate

            # Store landmark coordinates in list
            right_eye_coords.append((x, y))

            cv2.circle(image, (x, y), 3, (0, 0, 255), -1)  # Red color for right eye
            cv2.putText(image, f"P{i+1}", (x, y), FONT, ldmk_font_size, (0, 0, 255), 1)


            # Show landmark coordinates
            #cv2.putText(image, f"P{i+1}: ({x}, {y})", (info_right_eye_x, info_ldmk_y+text_y_offset), FONT, info_font_size, (0, 0, 255), 2)
            #text_y_offset += 15
        
           ear_right, A_right, B_right, C_right = calc_ear(right_eye_coords)
           
           # Display the right eye features
           text_y_offset += 15
           cv2.putText(image, f"EAR: {ear_right:.2f}", (info_left_eye_x, info_ldmk_y+text_y_offset), FONT, info_font_size, (0, 0, 255), 2)
           text_y_offset += 15
           cv2.putText(image, f"P2-P6: {A_right:.2f}", (info_left_eye_x, info_ldmk_y+text_y_offset), FONT, info_font_size, (0, 0, 255), 2)
           text_y_offset += 15
           cv2.putText(image, f"P3-P5: {B_right:.2f}", (info_left_eye_x, info_ldmk_y+text_y_offset), FONT, info_font_size, (0, 0, 255), 2)
           text_y_offset += 15
           cv2.putText(image, f"P1-P4: {C_right:.2f}", (info_left_eye_x, info_ldmk_y+text_y_offset), FONT, info_font_size, (0, 0, 255), 2)
           text_y_offset += 30

           #Average EAR
           ear_avg = (ear_left + ear_right) / 2

           cv2.putText(image, f"EAR AVG: {ear_avg:.2f}", (info_left_eye_x, info_ldmk_y+text_y_offset), FONT, info_font_size, (0, 255, 255), 2)
           text_y_offset += 20

           eye_time_text_y = info_ldmk_y+text_y_offset
           
           ### MOUTH FEATURES###
           # Highlight specific landmarks for the mouth
           mouth_coords = []
           
           # Draw text background rectangle
           cv2.rectangle(image, (info_left_eye_x, info_ldmk_y+text_y_offset), (info_left_eye_x+135, info_ldmk_y+text_y_offset+155), text_background, -1)

           text_y_offset += 30
           cv2.putText(image, f"MOUTH FEATs", (info_left_eye_x, info_ldmk_y+text_y_offset), FONT, info_font_size, (255, 255, 255), 2)
           text_y_offset += 20
           cv2.putText(image, f"Mouth LMKs:", (info_left_eye_x, info_ldmk_y+text_y_offset), FONT, info_font_size, (255, 0, 0), 2)
           for i, idx in enumerate(mouth_indices):
            # Obtain x,y coordinates of mouth landmarks
            x = int(face_landmarks.landmark[idx].x * image.shape[1]) # Frame X coordinate
            y = int(face_landmarks.landmark[idx].y * image.shape[0]) # Frame Y coordinate

            # Store landmark coordinates in list
            mouth_coords.append((x, y))

            cv2.circle(image, (x, y), 3, (255, 0, 0), -1)  # Blue color for mouth
            cv2.putText(image, f"P{i+1}", (x, y), FONT, ldmk_font_size, (255, 0, 0), 1)

            # Show landmark coordinates
            #cv2.putText(image, f"P{i+1}: ({x}, {y})", (info_mouth_x, info_ldmk_y+text_y_offset), FONT, info_font_size, (255, 0, 0), 2)
            #text_y_offset += 15

           mar, A_mar, B_mar, C_mar, D_mar = calc_mar(mouth_coords)
           
           # Display mouth features
           text_y_offset += 15
           cv2.putText(image, f"MAR: {mar:.2f}", (info_left_eye_x, info_ldmk_y+text_y_offset), FONT, info_font_size, (255, 0, 0), 2)
           text_y_offset += 15
           cv2.putText(image, f"P2-P8: {A_mar:.2f}", (info_left_eye_x, info_ldmk_y+text_y_offset), FONT, info_font_size, (255, 0, 0), 2)
           text_y_offset += 15
           cv2.putText(image, f"P3-P7: {B_mar:.2f}", (info_left_eye_x, info_ldmk_y+text_y_offset), FONT, info_font_size, (255, 0, 0), 2)
           text_y_offset += 15
           cv2.putText(image, f"P4-P6: {C_mar:.2f}", (info_left_eye_x, info_ldmk_y+text_y_offset), FONT, info_font_size, (255, 0, 0), 2)
           text_y_offset += 15
           cv2.putText(image, f"P1-P5: {D_mar:.2f}", (info_left_eye_x, info_ldmk_y+text_y_offset), FONT, info_font_size, (255, 0, 0), 2)
           text_y_offset += 20
        
                
    
           #EAR Threshold for closed eyes
           if ear_avg < ear_threshold:
              if start_time is None:
                 start_time = time.time()
              else:
                 eyes_closed_time = time.time() - start_time
              if eyes_closed_time > 3:
                 cv2.putText(image, "CLOSED EYES ALERT!", (int(img_w/2)-200, int(img_h)-50), FONT, 1, (0, 0, 255), 2)
           else:
              start_time = None
              eyes_closed_time = 0
        
           # Display the time eyes are closed
           cv2.putText(image, f"Closed: {eyes_closed_time:.2f} s", (info_left_eye_x, eye_time_text_y), FONT, info_font_size, (0,165,255), 2)

           #MAR Threshold for closed eyes
           if mar > mar_threshold:
              if start_time_mouth is None:
                 start_time_mouth = time.time()
              else:
                 mouth_opened_time = time.time() - start_time_mouth
              #if mouth_opened_time > 3:
                 #cv2.putText(image, "OPEN MOUTH ALERT!", (int(img_w/2)-200, int(img_h)-50), FONT, 1, (0, 0, 255), 2)
           else:
              start_time_mouth = None
              mouth_opened_time = 0
        
           # Display the time mouth is open
           cv2.putText(image, f"Opened: {mouth_opened_time:.2f} s", (info_left_eye_x, info_ldmk_y+text_y_offset), FONT, info_font_size, (0, 165, 255), 2)

           ########### END of Facial feature recogition #################

       # Store data in lists
       timestamps.append(time.time())
       face_detections.append(face_detected)
       ear_left_val.append(ear_left)
       ear_right_val.append(ear_right)
       ear_avg_val.append(ear_avg)
       mar_val.append(mar)
       head_rot_x.append(x_rot)
       head_rot_y.append(y_rot)
       head_rot_z.append(z_rot)
       #print(face_detected,ear_left,ear_right,ear_avg,mar,x_rot,y_rot,z_rot)


    elif mode == 1:
       #print('Mode 2: Camera Image')
       image = cv2.flip(image, 0)
       
       # Display Image Display Mode
       cv2.putText(image, "Image Display", (10, 30), FONT, 1, (255, 255, 255), 2)


       # Display the frame without processing
       # Check for 'r' key to toggle recording
       if cv2.waitKey(1) & 0xFF == ord('r'):
        print('Recording Mode Change')
        recording = not recording
       if recording:
          cv2.circle(image, (270, 20), 10, (0, 0, 255), -1)  # Red circle
          out.write(image)  # Write frame to video file
              
    # Flip the image horizontally for a selfie-view display.
    #cv2.imshow('MediaPipe Face Mesh', cv2.flip(image, 1))

    cv2.imshow('Facial Features', image)

    if cv2.waitKey(5) & 0xFF == 27: #Press Esc to end live camera feed
      break

cap.release()
out.release()
cv2.destroyAllWindows()


####### STORE DATA IN CSV #########
""" # Specify the filename
csv_filename = "facial_feature_data.csv"

# Open the file in write mode
with open(csv_filename, mode='w', newline='') as file:
    writer = csv.writer(file)

    # Write the header
    writer.writerow(['Timestamp', 'Face Detected', 'EAR Left', 'EAR Right', 'EAR Avg', 'MAR', 'Head Rot X', 'Head Rot Y', 'Head Rot Z'])

    # Write the data
    for i in range(len(timestamps)):
        writer.writerow([timestamps[i], face_detections[i], ear_left_val[i], ear_right_val[i], ear_avg_val[i], mar_val[i], head_rot_x[i], head_rot_y[i], head_rot_z[i]]) """