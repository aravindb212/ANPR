import ast
import cv2
import numpy as np
import pandas as pd

def draw_border(img, top_left, bottom_right, color=(0, 255, 0), thickness=10, line_length_x=200, line_length_y=200):
    x1, y1 = top_left
    x2, y2 = bottom_right

    # Draw borders around the car
    cv2.line(img, (x1, y1), (x1, y1 + line_length_y), color, thickness)
    cv2.line(img, (x1, y1), (x1 + line_length_x, y1), color, thickness)
    cv2.line(img, (x1, y2), (x1, y2 - line_length_y), color, thickness)
    cv2.line(img, (x1, y2), (x1 + line_length_x, y2), color, thickness)
    cv2.line(img, (x2, y1), (x2 - line_length_x, y1), color, thickness)
    cv2.line(img, (x2, y1), (x2, y1 + line_length_y), color, thickness)
    cv2.line(img, (x2, y2), (x2, y2 - line_length_y), color, thickness)
    cv2.line(img, (x2, y2), (x2 - line_length_x, y2), color, thickness)

    return img

# Read the interpolated results from CSV
results = pd.read_csv('./test_interpolated.csv')

# Load video
video_path = 'testUK.mp4'
cap = cv2.VideoCapture(video_path)

# Specify the codec and other parameters for the output video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter('./final_out.mp4', fourcc, fps, (width, height))
print('fps :',fps)
# Initialize a dictionary to store license plate details for each car ID
license_plate = {}

# Iterate over each unique car ID in the results
for car_id in np.unique(results['car_id']):
    # Find the maximum license plate score for the current car ID
    max_score = np.amax(results[results['car_id'] == car_id]['license_number_score'])
    
    # Store license plate information for the car ID
    license_plate[car_id] = {
        'license_crop': None,
        'license_plate_number': results[(results['car_id'] == car_id) &
                                        (results['license_number_score'] == max_score)]['license_number'].iloc[0]
    }
    
    # Set the video frame to the one where the highest score for the license plate was detected
    cap.set(cv2.CAP_PROP_POS_FRAMES, results[(results['car_id'] == car_id) &
                                             (results['license_number_score'] == max_score)]['frame_num'].iloc[0])
    
    # Read the frame
    ret, frame = cap.read()
    
    # Extract license plate bounding box coordinates
    x1, y1, x2, y2 = ast.literal_eval(results[(results['car_id'] == car_id) &
                                              (results['license_number_score'] == max_score)]['license_plate_bbox'].iloc[0].replace('[ ', '[').replace('   ', ' ').replace('  ', ' ').replace(' ', ','))
    
    # Crop and resize the license plate from the frame
    license_crop = frame[int(y1):int(y2), int(x1):int(x2), :]
    license_crop = cv2.resize(license_crop, (int((x2 - x1) * 400 / (y2 - y1)), 400))
    
    # Store the cropped license plate image
    license_plate[car_id]['license_crop'] = license_crop

# Initialize frame number
frame_num = -1

# Reset video capture to the beginning
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

# Read frames from the video
ret = True
while ret:
    ret, frame = cap.read()
    frame_num += 1
    if ret:
        # Filter results dataframe for the current frame number
        df_ = results[results['frame_num'] == frame_num]
        
        # Iterate over each row in the filtered dataframe
        for row_indx in range(len(df_)):
            # Draw bounding box around the car
            car_x1, car_y1, car_x2, car_y2 = ast.literal_eval(df_.iloc[row_indx]['car_bbox'].replace('[ ', '[').replace('   ', ' ').replace('  ', ' ').replace(' ', ','))
            draw_border(frame, (int(car_x1), int(car_y1)), (int(car_x2), int(car_y2)), (0, 255, 0), 25,
                        line_length_x=200, line_length_y=200)

            # Draw bounding box around the license plate
            x1, y1, x2, y2 = ast.literal_eval(df_.iloc[row_indx]['license_plate_bbox'].replace('[ ', '[').replace('   ', ' ').replace('  ', ' ').replace(' ', ','))
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 12)

            try:
                # Overlay cropped license plate onto the frame
                license_crop = license_plate[df_.iloc[row_indx]['car_id']]['license_crop']
                H, W, _ = license_crop.shape
                
                frame[int(car_y1) - H - 100:int(car_y1) - 100,
                      int((car_x2 + car_x1 - W) / 2):int((car_x2 + car_x1 + W) / 2), :] = license_crop

                # Add white space below the license plate for text overlay
                frame[int(car_y1) - H - 400:int(car_y1) - H - 100,
                      int((car_x2 + car_x1 - W) / 2):int((car_x2 + car_x1 + W) / 2), :] = (255, 255, 255)

                # Add license plate number text on the frame
                (text_width, text_height), _ = cv2.getTextSize(
                    license_plate[df_.iloc[row_indx]['car_id']]['license_plate_number'],
                    cv2.FONT_HERSHEY_SIMPLEX,
                    4.3,
                    17)

                cv2.putText(frame,
                            license_plate[df_.iloc[row_indx]['car_id']]['license_plate_number'],
                            (int((car_x2 + car_x1 - text_width) / 2), int(car_y1 - H - 250 + (text_height / 2))),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            4.3,
                            (0, 0, 0),
                            17)

            except Exception as e:
                print(f"Error processing frame {frame_num}: {e}")

        # Write the processed frame to the output video
        out.write(frame)
        frame = cv2.resize(frame, (1280, 720))

        # Display the processed frame (uncomment if needed for debugging)
        # cv2.imshow('frame', frame)
        # cv2.waitKey(0)

# Release resources
out.release()
cap.release()
