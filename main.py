from ultralytics import YOLO  # Import the YOLO class from the ultralytics library
import cv2  # Import OpenCV for video processing
from sort.sort import *  # Import everything from the sort.sort module (for object tracking)
import numpy as np  # Import numpy for numerical operations
from utils import Get_Car, Read_License_Plate, Write_CSV  # Import custom utility functions for getting cars, reading license plates, and writing CSVs
import pandas as pd  # Import pandas for data manipulation (although it's not used in this snippet)

# Initialize the YOLO model with a pre-trained COCO model to detect vehicles
model = YOLO('yolov8n.pt')
# Initialize another YOLO model for detecting license plates
LP_detector = YOLO('license_plate_detector.pt')
 
 
# Open the video file
cap = cv2.VideoCapture('testUK.mp4')
# Define the class IDs for vehicles in the COCO dataset
vehicles = [2, 3, 5, 7]

frame_num = -1  # Initialize frame number
ret = True  # Initialize the return value for the video read loop
mot_tracker = Sort()  # Initialize the SORT tracker
results = {}  # Initialize a dictionary to store the results

while ret:  # Loop while the video read is successful
    frame_num += 1  # Increment the frame number
    ret, frame = cap.read()  # Read a frame from the video

    if ret :  # If the frame read is successful
        results[frame_num] = {}  # Initialize a dictionary for the current frame results
        
        # Detect vehicles in the frame
        detections = model(frame)[0]  # Get the detection results from the YOLO model
        detections_ = []  # Initialize a list to store filtered detections
        
        for detection in detections.boxes.data.tolist():  # Loop through each detection
            x1, y1, x2, y2, score, class_id = detection  # Unpack the detection details
            if int(class_id) in vehicles:  # If the detected class is a vehicle
                detections_.append([x1, y1, x2, y2, score])  # Add the detection to the list
        if len(detections_) == 0:  # If no vehicles are detected, continue to the next frame
            continue

        # Track vehicles in the video
        track_ids = mot_tracker.update(np.asarray(detections_))  # Update the tracker with the new detections
        print("result track ids:", track_ids)  # Print the tracked IDs
        
        # Detect license plates in the frame
        LP = LP_detector(frame)[0]  # Get the license plate detection results
        for lp in LP.boxes.data.tolist():  # Loop through each license plate detection
            x1, y1, x2, y2, score, class_id = lp  # Unpack the detection details
            print("License Plate Detection:", lp)  # Print the license plate detection details

            # Assign license plate to a vehicle
            if track_ids.any():  # If there are tracked vehicles
                xcar1, ycar1, xcar2, ycar2, car_id = Get_Car(lp, track_ids)  # Get the car details for the license plate
                print(Get_Car(lp, track_ids))  # Print the car details
                
                if car_id != -1:  # If a car is found for the license plate
                    # Crop the license plate from the frame
                    lp_crop = frame[int(y1): int(y2), int(x1): int(x2), :]

                    # Process the cropped license plate
                    lp_crop_gray = cv2.cvtColor(lp_crop, cv2.COLOR_BGR2GRAY)
                    lp_crop_gray = cv2.fastNlMeansDenoising(lp_crop_gray, None, 30, 7, 21)
                    lp_crop_gray = cv2.equalizeHist(lp_crop_gray)
                    _, lp_crop_tsh = cv2.threshold(lp_crop_gray, 90, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    
                    # Read the license plate text
                    lp_text, lp_text_score = Read_License_Plate(lp_crop_tsh)  # Read the license plate text
                    print(Read_License_Plate(lp_crop_tsh))  # Print the read text and score

                    if lp_text is not None:  # If a license plate text is detected
                        # Store the detection results in the results dictionary
                        results[frame_num][str(car_id)] = {
                            'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                            'license_plate': {
                                'bbox': [x1, y1, x2, y2],
                                'text': lp_text,
                                'bbox_score': score,
                                'text_score': lp_text_score
                            }
                        }
                        print('results_final', results[frame_num][str(car_id)])  # Print the final results for the frame

# Write the results to a CSV file
Write_CSV(results, 'output.csv')
