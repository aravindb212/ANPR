import string  # Import string module for string operations
import easyocr  # Import EasyOCR for OCR text recognition
import cv2  # Import OpenCV for image processing
import csv  # Import CSV for CSV file operations
import pytesseract

# Set Tesseract path (adjust as necessary)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Initialize the OCR reader with English language and GPU support
reader = easyocr.Reader(['en'], gpu=True)

# Mapping dictionaries for character conversions
dict_char_to_int = {
    'O': '0', 'I': '1', 'L': '1', 'J': '1', 'Z': '2',  # Characters to integers
    'S': '5', 'G': '6', 'B': '8', 'A': '4', 'Q': '0',
    'P': '7', 'Y': '7', '?': '7'
}


dict_int_to_char = {y: x for x, y in dict_char_to_int.items()}  # Reverse mapping dictionary


def Write_CSV(results, output_path):
    """
    Write the results to a CSV file.

    Args:
        results (dict): Dictionary containing the results.
        output_path (str): Path to the output CSV file.
    """
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['frame_num', 'car_id', 'car_bbox', 'license_plate_bbox',
                         'license_plate_bbox_score', 'license_number', 'license_number_score'])

        for frame_num in results.keys():
            for car_id in results[frame_num].keys():
                if 'car' in results[frame_num][car_id].keys() and \
                   'license_plate' in results[frame_num][car_id].keys() and \
                   'text' in results[frame_num][car_id]['license_plate'].keys():
                    writer.writerow([
                        frame_num,
                        car_id,
                        '[{} {} {} {}]'.format(*results[frame_num][car_id]['car']['bbox']),
                        '[{} {} {} {}]'.format(*results[frame_num][car_id]['license_plate']['bbox']),
                        results[frame_num][car_id]['license_plate']['bbox_score'],
                        results[frame_num][car_id]['license_plate']['text'],
                        results[frame_num][car_id]['license_plate']['text_score']
                    ])


def Read_License_Plate(lp_crop):
    """
    Read and format the license plate text from the cropped image.

    Args:
        lp_crop (numpy.ndarray): Cropped license plate image.

    Returns:
        tuple: (Formatted license plate text, Confidence score)
    """
    
    detections = reader.readtext(lp_crop)  # Perform OCR on the cropped image
    for detection in detections:
        bbox, text, score = detection
        text = text.upper().replace(' ', '')  # Clean up OCR text
   
        if License_Compiles_Format(text):  # Check if the license plate format complies
                     return Format_License(text),score  # Return formatted text and confidence score
    
    # Return the best text and its score
    return None, None


     

def Format_License(text):
    """
    Format the license plate text by converting characters using the mapping dictionaries.

    Args:
        text (str): License plate text.

    Returns:
        str: Formatted license plate text.
    """
    license_plate_ = ''
    mapping = {0: dict_int_to_char, 1: dict_int_to_char,
               2: dict_char_to_int, 3: dict_char_to_int,
               4: dict_int_to_char, 5: dict_int_to_char,
               6: dict_int_to_char}

    for j in range(7):  # Iterate over characters in the license plate (fixed length 7)
        if text[j] in mapping[j].keys():
            license_plate_ += mapping[j][text[j]]
        else:
            license_plate_ += text[j]

    return license_plate_


def License_Compiles_Format(text):
    """
    Check if the license plate text complies with the required format.

    Args:
        text (str): License plate text.

    Returns:
        bool: True if the license plate complies with the format, False otherwise.
    """
    if len(text) != 7:  # Check if the length of the text is exactly 7 characters
        return False

    if (text[0] in string.ascii_uppercase or text[0] in dict_int_to_char.keys()) and \
       (text[1] in string.ascii_uppercase or text[1] in dict_int_to_char.keys()) and \
       (text[2] in string.digits or text[2] in dict_char_to_int.keys()) and \
       (text[3] in string.digits or text[3] in dict_char_to_int.keys()) and \
       (text[4] in string.ascii_uppercase or text[4] in dict_int_to_char.keys()) and \
       (text[5] in string.ascii_uppercase or text[5] in dict_int_to_char.keys()) and \
       (text[6] in string.ascii_uppercase or text[6] in dict_int_to_char.keys()):
        return True

    return False
def Get_Car(lp, track_ids):
    """
    Match license plate detection to a tracked car based on overlap.

    Args:
        lp (list): License plate detection [x1, y1, x2, y2, score, class_id].
        track_ids (numpy.ndarray): Array of tracked objects [x1, y1, x2, y2, id].

    Returns:
        tuple: Coordinates and ID of the matched car if found, otherwise (-1, -1, -1, -1, -1).
    """
    x1, y1, x2, y2, score, class_id = lp

    for car in track_ids:
        xcar1, ycar1, xcar2, ycar2, car_id = car

        # Check if the license plate bounding box overlaps with the tracked car bounding box
        if x1 > xcar1 and x2 < xcar2 and y1 > ycar1 and y2 < ycar2:
            return car

    return -1, -1, -1, -1, -1  # Return if no match found

   
