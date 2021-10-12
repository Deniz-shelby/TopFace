import numpy as np
import math

def get_landmarks(frame, results,landmark_number_list):
    shape = frame.shape
    landmark_list = []
    for landmark_number in landmark_number_list:
        x = results.face_landmarks.landmark[int(landmark_number)].x
        y = results.face_landmarks.landmark[int(landmark_number)].y
        relative_x = int(x * shape[1])
        relative_y = int(y * shape[0])
        landmark_list.append([landmark_number,relative_x, relative_y])
    return landmark_list

def get_center(iris_list):
    x = (iris_list[0][0]-iris_list[1][0])/2 + iris_list[1][0]
    y = (iris_list[0][1]-iris_list[1][1])/2 + iris_list[1][1]
    center = (int(x),int(y))
    return center

def calculate_angle(first_point,second_point):
    b = np.array(first_point)
    c = np.array(second_point)
    a = [b[0],c[1]]
    a = np.array(a)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0]) #[1] = y //[0] = x
    angle = radians*180.0/np.pi
    return angle

def calculate_distance_position_with_angle(distance,angle):
    ankathet = np.cos(angle)*distance
    countered = np.sin(angle)*distance
    return [int(ankathet),int(countered)]

def calculate_distance_position(distance):
    ankathet = distance/math.sqrt(2)
    countered = distance/math.sqrt(2)
    return [int(ankathet),int(countered)]

def midpoint(point1 ,point2):
    return int((point1[0] + point2[0])/2), int((point1[1] + point2[1])/2)

def euclidean_distance(point1 , point2):
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def get_blink_ratio(eye_points_1,eye_points_2):
    
    #loading all the required points
    corner_left_1  = (eye_points_1[0][0], eye_points_1[0][1])
                    
    corner_right_1 = (eye_points_1[3][0], eye_points_1[3][1])
    
    center_top_1    = midpoint(eye_points_1[1], eye_points_1[2])
    center_bottom_1 = midpoint(eye_points_1[5], eye_points_1[4])

    #calculating distance
    horizontal_length_1 = euclidean_distance(corner_left_1,corner_right_1)
    vertical_length_1 = euclidean_distance(center_top_1,center_bottom_1)

    ratio_1 = horizontal_length_1 / vertical_length_1

        #loading all the required points
    corner_left_2  = (eye_points_2[0][0], eye_points_2[0][1])
                    
    corner_right_2 = (eye_points_2[3][0], eye_points_2[3][1])
    
    center_top_2    = midpoint(eye_points_2[1], eye_points_2[2])
    center_bottom_2 = midpoint(eye_points_2[5], eye_points_2[4])

    #calculating distance
    horizontal_length_2 = euclidean_distance(corner_left_2,corner_right_2)
    vertical_length_2 = euclidean_distance(center_top_2,center_bottom_2)

    ratio_2 = horizontal_length_2 / vertical_length_2

    return [ratio_1,ratio_2]