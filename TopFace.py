import mediapipe as mp
import cv2
import numpy as np
import math
import csv
import os
import pandas as pd
from tensorflow import keras
import pickle
from functions_for_topface import get_landmarks, get_center, calculate_angle, calculate_distance_position_with_angle, calculate_distance_position, midpoint, euclidean_distance, get_blink_ratio


mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic # Mediapipe Solutions

# Load the models
emotion_model = keras.models.load_model('emotion_model')
emotion_dict = {0: "Not Happy", 1: "Not Happy", 2: "Not Happy",
                3: "Happy", 4: "Not Happy", 5: "Not Happy", 6: "Not Happy"}

with open('body_language.pkl', 'rb') as f:
    model = pickle.load(f)

# start application

cap = cv2.VideoCapture(0)
cap.set(3,1920) #width
cap.set(4,1282) #height
cap.set(10,10) #brightness

#1920 x 1282 

count = 0
hand_raised_count = 0
prediction = []
emotion_prediction_list = []
character_change = ['Next','Next','Next','Next','Next','Next']
hand_raised = ['Raised Hand','Raised Hand','Raised Hand','Raised Hand','Raised Hand','Raised Hand']
###########  read_background
background_frame = cv2.imread('background_bamboo_big.jpg')

# Initiate holistic model
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
        # If loading a video, use 'break' instead of 'continue'.
            continue
        
        # Recolor Feed
        frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
        frame.flags.writeable = False

        # Predict Emotion
        bounding_box = cv2.CascadeClassifier(
            'Haarcascades/haarcascade_frontalface_default.xml')
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        num_faces = bounding_box.detectMultiScale(
            gray_frame, scaleFactor=1.3, minNeighbors=5)        
        
        for (x, y, w, h) in num_faces:
            roi_gray_frame = gray_frame[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(
                cv2.resize(roi_gray_frame, (48, 48)), -1), 0)
            emotion_prediction = emotion_model.predict(cropped_img)
            maxindex = int(np.argmax(emotion_prediction))
        # Make Emotion Detections
        emotion_prediction = str(emotion_dict[maxindex])
        emotion_prediction_list.append(emotion_prediction)

        results = holistic.process(frame)
        
        # Recolor frame back to BGR for rendering
        frame.flags.writeable = True   
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        raw_frame = frame.copy()
        

        
        # Export coordinates
        
        try:   
            try:
                # Extract Left Hand landmarks
                left_hand = results.left_hand_landmarks.landmark
                left_hand_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in left_hand]).flatten())
            
            except:
                left_hand_row = list(np.array([[0, 0, 0, landmark.visibility] for landmark in left_hand]).flatten())

            try:
                # Extract Right Hand landmarks
                right_hand = results.right_hand_landmarks.landmark
                right_hand_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in right_hand]).flatten())

            except:
                right_hand_row = list(np.array([[0, 0, 0, 0] for landmark in right_hand]).flatten())


            # Concate rows
            row = left_hand_row+right_hand_row
            
            # Make Detection
            X = pd.DataFrame([row])
            body_language_class = model.predict(X)[0]
            body_language_prob = model.predict_proba(X)[0]
            
            prediction.append(body_language_class)
            if ((prediction[-6:] == character_change) and (prediction[-7] != 'Next')):
                count += 1
            
            if ((prediction[-6:] == hand_raised) and (prediction[-7] != 'Raised Hand')):
                hand_raised_count += 1
            print(hand_raised_count)
            happy_counter = emotion_prediction_list.count('Happy')
            #print(happy_counter)
            happy_percentage = round((happy_counter/len(emotion_prediction_list))*100,2)



                    ######################## ADD BACKGROUND ################################
        
            frame = background_frame.copy() ## h*w same as capture screen size


            # # Get status box
            cv2.rectangle(frame, (0,0), (400, 200), (255, 0, 255), -1)
            
            cv2.rectangle(frame, (0,200), (400, 400), (255, 0, 255), -1)

            cv2.rectangle(frame, (0,1070), (480, 1300), (255, 0, 255), -1)

            cv2.rectangle(frame, (1620,0), (1920, 200), (255, 0, 255), -1)

            
                
            if body_language_class.split(' ')[0] != 'Neutral':
            # Display Class
                cv2.putText(frame, 'CLASS'
                            , (25,60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(frame, body_language_class.split(' ')[0]
                            , (70,160), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 5, cv2.LINE_AA)
            
            if emotion_prediction == 'Happy':
            # Display Class
                cv2.putText(frame, 'Emotion'
                            , (25,240), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(frame, emotion_prediction
                            , (70,340), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 5, cv2.LINE_AA)
        
        
            cv2.putText(frame, 'Happy %'
                            , (1640,60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(frame, str(happy_percentage)
                            , (1640,160), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 5, cv2.LINE_AA)

            cv2.putText(frame, 'Raised Hands'
                            , (10,1150), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0),3, cv2.LINE_AA)
            cv2.putText(frame, str(hand_raised_count)
                            , (170,1258), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 5, cv2.LINE_AA)
        
        except:
            pass
        ###################### GET LANDMARK NUMBERS ###############################
        top_mouth_position_number = [0]
        top_inner_mouth_position_number = [13]

        bottom_mouth_position_number = [17]
        bottom_inner_mouth_position_number = [14]

        # face
        center_face_position_number = [10,152]

        # nose
        center_nose_position_number = [4]

        #nose_mouth
        bird_nose_mouth_position_number = [36,266]

        #eyes
        left_eye_number_list = [159,145]
        left_eye_ratio_number_list = [35,160,158,133,153,144]
        
        right_eye_number_list = [386,374]
        right_eye_ratio_number_list = [236,387,385,362,380,373]

        #ears
        left_ear_position_number = [103]
        right_ear_position_number = [332]

        #wings
        left_wing_position_number = [234]
        right_wing_position_number = [454]

        #alignment
        alignment_points_number = [8,4]

        # size of the face
        size_points_number = [10,152]
         
        if results.face_landmarks:
            #face
            center_face_position = get_landmarks(frame, results,center_face_position_number)
            #nose
            center_nose_position = get_landmarks(frame, results,center_nose_position_number)
            #nose_mouth
            bird_nose_mouth_position = get_landmarks(frame, results,bird_nose_mouth_position_number)
            #mouth
            top_mouth_position = get_landmarks(frame, results,top_mouth_position_number)
            top_inner_mouth_position = get_landmarks(frame, results,top_inner_mouth_position_number)
            bottom_mouth_position = get_landmarks(frame, results,bottom_mouth_position_number)
            bottom_inner_mouth_position = get_landmarks(frame, results,bottom_inner_mouth_position_number)
            #ears
            left_ear_position_landmark = get_landmarks(frame, results,left_ear_position_number)
            right_ear_position_landmark = get_landmarks(frame, results,right_ear_position_number)
            #wings
            left_wing_position_landmark = get_landmarks(frame, results,left_wing_position_number)
            right_wing_position_landmark = get_landmarks(frame, results,right_wing_position_number)
            #eyes
            left_eye_list_of_landmarks = get_landmarks(frame, results,left_eye_number_list)
            left_eye_ratio_list_of_landmarks = get_landmarks(frame, results,left_eye_ratio_number_list)
            right_eye_list_of_landmarks = get_landmarks(frame, results,right_eye_number_list)
            right_eye_ratio_list_of_landmarks = get_landmarks(frame, results,right_eye_ratio_number_list)


            alignment_points_landmarks = get_landmarks(frame, results,alignment_points_number)

            size_points_landmarks = get_landmarks(frame, results,size_points_number)

            ### get landmarks
            left_eye_position_list = []
            for landmark in left_eye_list_of_landmarks:
                left_eye_position_list.append([landmark[1],landmark[2]])

            left_eye_ratio_position_list = []
            for landmark in left_eye_ratio_list_of_landmarks:
                left_eye_ratio_position_list.append([landmark[1],landmark[2]])
                
            right_eye_position_list = []
            for landmark in right_eye_list_of_landmarks:
                right_eye_position_list.append([landmark[1],landmark[2]])

            right_eye_ratio_position_list = []
            for landmark in right_eye_ratio_list_of_landmarks:
                right_eye_ratio_position_list.append([landmark[1],landmark[2]])
            
            center_face_position_list = []
            for landmark in center_face_position:
                center_face_position_list.append([landmark[1],landmark[2]])

            bird_nose_mouth_position_list = []
            for landmark in bird_nose_mouth_position:
                bird_nose_mouth_position_list.append([landmark[1],landmark[2]])

            alignment_points_position_list = []
            for alignment_points_landmark in alignment_points_landmarks:
                alignment_points_position_list.append([alignment_points_landmark[1],alignment_points_landmark[2]])

            size_points_position_list = []
            for size_points_landmark in size_points_landmarks:
                size_points_position_list.append([size_points_landmark[1],size_points_landmark[2]])
            
            
            # ear positon
            left_ear_position = [left_ear_position_landmark[0][1]-17,left_ear_position_landmark[0][2]-17]
            right_ear_position = [right_ear_position_landmark[0][1]+17,right_ear_position_landmark[0][2]-17]

            # # wing positon
            # left_wing_position = [left_wing_position_landmark[0][1]-17,left_wing_position_landmark[0][2]-17]
            # right_wing_position = [right_wing_position_landmark[0][1]+17,right_wing_position_landmark[0][2]-17]

            #mouth position
            top_mouth_position = [top_mouth_position[0][1],top_mouth_position[0][2]]
            top_inner_mouth_position = [top_inner_mouth_position[0][1],top_inner_mouth_position[0][2]]
            bottom_mouth_position = [bottom_mouth_position[0][1],bottom_mouth_position[0][2]]
            bottom_inner_mouth_position = [bottom_inner_mouth_position[0][1],bottom_inner_mouth_position[0][2]]
            
            # getting angle from alignment points
            angle = calculate_angle(alignment_points_position_list[0],alignment_points_position_list[1])
            #print(angle)

            # sizing
            size = leng=int(math.hypot(size_points_position_list[1][0] - size_points_position_list[0][0],size_points_position_list[1][1] - size_points_position_list[0][1])*0.65)
            #print(size)

            ear_x_size = int(size*0.35)
            ear_y_size = int(size*0.45)

            eye_big_x_size = int(size*0.40)
            eye_big_y_size = int(size*0.30)

            wing_x_size = int(size*0.25)
            wing_y_size = int(size*1.10)
            
            eye_small_size = int(size*0.15)
            bird_eye_small_size = int(size*0.25)
            eye_iris_size = int(size*0.07)
            bird_eye_iris_size = int(size*0.14)

            nose_big_size = int(size*0.2)
            nose_small_size = int(size*0.1)

            eye_to_big_eye_distance = int(size*0.14)

            #########
            mouth_lip_distance = int(size*0.15)
            

            inner_mouth_distance = bottom_inner_mouth_position[1]-top_inner_mouth_position[1]
            if inner_mouth_distance > 0:
                inner_mouth_distance = inner_mouth_distance*3
            else:
                inner_mouth_distance = 0
            
            mouth_starting_angle = 0 +angle*0.5
            mouth_ending_angle = 180 +angle*0.5

            #eyes
            center_of_left_eye = get_center(left_eye_position_list)
            center_of_right_eye = get_center(right_eye_position_list)

            # eye_angle = 45-angle
            # distance_eye_position = calculate_distance_position_with_angle(eye_to_big_eye_distance,eye_angle)
            distance_eye_position = calculate_distance_position(eye_to_big_eye_distance)

            left_eye_big = [center_of_left_eye[0]-distance_eye_position[0],center_of_left_eye[1]+distance_eye_position[1]]
            right_eye_big = [center_of_right_eye[0]+distance_eye_position[0],center_of_right_eye[1]+distance_eye_position[1]]



            ## Bird mouth_nose
            mouth_nose_pts = np.array([bird_nose_mouth_position_list[0],bird_nose_mouth_position_list[1],top_inner_mouth_position])
            mouth_nose_pts = mouth_nose_pts.reshape((-1,1,2))
            bird_mouth_opening = [top_inner_mouth_position[0],top_inner_mouth_position[1]+inner_mouth_distance]
            mouth_nose_pts_opening = np.array([bird_nose_mouth_position_list[0],bird_nose_mouth_position_list[1],bird_mouth_opening])
            mouth_nose_pts_opening = mouth_nose_pts_opening.reshape((-1,1,2))

            ########################### ROTATION LOGIC
            

            ########################### Blinking
            #### combine both 
            eye_ratio = get_blink_ratio(left_eye_ratio_position_list,right_eye_ratio_position_list)

            BLINK_RATIO_THRESHOLD = 6.7
            #print(left_eye_ratio )
            if eye_ratio[0] > BLINK_RATIO_THRESHOLD:
            #Blink detected! Do Something!
                # cv2.putText(frame,"BLINKING",(10,50), cv2.FONT_HERSHEY_SIMPLEX,
                #             2,(255,255,255),2,cv2.LINE_AA)
                left_eye_closing_angels = 0
            else:
                left_eye_closing_angels = 360

            if eye_ratio[1] > BLINK_RATIO_THRESHOLD:
                right_eye_closing_angels = 0
            else:
                right_eye_closing_angels = 360

            center_of_face = get_center(center_face_position_list)    

            ##########


            

            

            ############################### DRAWING ###############################
            
            ############################### Character Logic ###############################
            
            if count % 2 == 0:
                character = 'panda'
            else:
                character = 'bird'
            if character == 'panda':
                
                 ###################### draw left ear
                cv2.ellipse(frame,(left_ear_position[0],left_ear_position[1]),(ear_x_size,ear_y_size),angle+45,0,360,(0,0,0),-1)
                

                ###################### draw right ear
                cv2.ellipse(frame,(right_ear_position[0],right_ear_position[1]),(ear_x_size,ear_y_size),angle-45,0,360,(0,0,0),-1)


                ###################### draw face-
                cv2.circle(frame, center_of_face, size, (255,255,255), -1)

                ###################### draw nose
                cv2.ellipse(frame,(center_nose_position[0][1],center_nose_position[0][2]) , (nose_big_size,nose_small_size), angle, 0, 360, (0,0,0), -1)

                
                ###################### draw left eye

                cv2.ellipse(frame,(center_of_left_eye[0]-distance_eye_position[0],center_of_left_eye[1]+distance_eye_position[1]) , (eye_big_x_size ,eye_big_y_size), angle-45, 0, 360, (0,0,0), -1)

                cv2.circle(frame,(center_of_left_eye[0],center_of_left_eye[1]),eye_small_size,(255,255,255),-1)
                cv2.ellipse(frame,(center_of_left_eye[0],center_of_left_eye[1]),(eye_small_size,eye_small_size),angle,left_eye_closing_angels,360,(0,0,0),-1)
                cv2.circle(frame,center_of_left_eye,eye_iris_size,(0,0,0),-1)

                ###################### draw right eye
                cv2.ellipse(frame,(center_of_right_eye[0]+distance_eye_position[0],center_of_right_eye[1]+distance_eye_position[1]) , (eye_big_x_size ,eye_big_y_size), angle+45, 0, 360, (0,0,0), -1)
                cv2.circle(frame,(center_of_right_eye[0],center_of_right_eye[1]),eye_small_size,(255,255,255),-1)
                cv2.ellipse(frame,(center_of_right_eye[0],center_of_right_eye[1]),(eye_small_size,eye_small_size),angle,right_eye_closing_angels,360,(0,0,0),-1)
                cv2.circle(frame,center_of_right_eye,eye_iris_size,(0,0,0),-1)

                ###################### draw mouth
                cv2.ellipse(frame,( top_inner_mouth_position[0], top_inner_mouth_position[1]+mouth_lip_distance) , (inner_mouth_distance ,inner_mouth_distance), angle, 0, 180, (193,193,255), -1)
                
                if inner_mouth_distance > 5:
                    cv2.ellipse(frame,( top_inner_mouth_position[0], top_inner_mouth_position[1]+mouth_lip_distance) , (inner_mouth_distance ,inner_mouth_distance), angle, 0, 180, (0,0,0), 5)
                

                cv2.line(frame,(top_inner_mouth_position[0],top_inner_mouth_position[1]),(center_nose_position[0][1],center_nose_position[0][2]),(0,0,0),5)
                cv2.ellipse(frame,(top_inner_mouth_position[0]-mouth_lip_distance,top_inner_mouth_position[1]) , (mouth_lip_distance,mouth_lip_distance), angle, 0, mouth_ending_angle, (255,255,255), -1)
                cv2.ellipse(frame,(top_inner_mouth_position[0]-mouth_lip_distance,top_inner_mouth_position[1]) , (mouth_lip_distance,mouth_lip_distance), angle, 0, mouth_ending_angle, (0,0,0), 5)
                cv2.ellipse(frame,(top_inner_mouth_position[0]+mouth_lip_distance,top_inner_mouth_position[1]) , (mouth_lip_distance ,mouth_lip_distance), angle, mouth_starting_angle, 180, (255,255,255), 5)
                cv2.ellipse(frame,(top_inner_mouth_position[0]+mouth_lip_distance,top_inner_mouth_position[1]) , (mouth_lip_distance ,mouth_lip_distance), angle, mouth_starting_angle, 180, (0,0,0), 5)

            elif character == 'bird':
                ###################### draw wings
                cv2.ellipse(frame,(left_wing_position_landmark[0][1],left_wing_position_landmark[0][2]) , ( wing_x_size,wing_y_size), 70+angle, 0, 360, (0,173,238), -1)
                cv2.ellipse(frame,(left_wing_position_landmark[0][1],left_wing_position_landmark[0][2]) , ( wing_x_size,wing_y_size), 80+angle, 0, 360, (0,238,238), -1)
                cv2.ellipse(frame,(left_wing_position_landmark[0][1],left_wing_position_landmark[0][2]) , ( wing_x_size,wing_y_size), 90+angle, 0, 360, (0,173,238), -1)

                cv2.ellipse(frame,(right_wing_position_landmark[0][1],right_wing_position_landmark[0][2]) , ( wing_x_size,wing_y_size), -70+angle, 0, 360, (0,173,238), -1)
                cv2.ellipse(frame,(right_wing_position_landmark[0][1],right_wing_position_landmark[0][2]) , ( wing_x_size,wing_y_size), -80+angle, 0, 360, (0,238,238), -1)
                cv2.ellipse(frame,(right_wing_position_landmark[0][1],right_wing_position_landmark[0][2]) , ( wing_x_size,wing_y_size), -90+angle, 0, 360, (0,173,238), -1)
                ###################### draw face-
                cv2.circle(frame, center_of_face, size, (0,238,238), -1)

                ###################### draw left eye
                
                cv2.circle(frame,(center_of_left_eye[0],center_of_left_eye[1]),bird_eye_small_size,(255,255,255),-1)
                cv2.circle(frame,center_of_left_eye,bird_eye_iris_size,(0,0,0),-1)
                cv2.ellipse(frame,(center_of_left_eye[0],center_of_left_eye[1]),(bird_eye_small_size,bird_eye_small_size),angle,left_eye_closing_angels,360,(0,238,238),-1)

                ###################### draw right eye
                
                cv2.circle(frame,(center_of_right_eye[0],center_of_right_eye[1]),bird_eye_small_size,(255,255,255),-1)
                cv2.circle(frame,center_of_right_eye,bird_eye_iris_size,(0,0,0),-1)
                cv2.ellipse(frame,(center_of_right_eye[0],center_of_right_eye[1]),(bird_eye_small_size,bird_eye_small_size),angle,right_eye_closing_angels,360,(0,238,238),-1)

                ###################### draw mouth_nose
                cv2.fillPoly(frame,[mouth_nose_pts_opening],(193,193,255))
                cv2.polylines(frame,[mouth_nose_pts_opening],True,(0,129,255),3)
                cv2.polylines(frame,[mouth_nose_pts],True,(0,129,255),5)
                cv2.fillPoly(frame,[mouth_nose_pts],(0,129,255))
                


        cv2.imshow('TopFace', frame)
        cv2.imshow('Raw Feed',raw_frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()