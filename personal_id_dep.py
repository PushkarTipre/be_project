from flask import Flask, jsonify, request
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
tf.get_logger().setLevel('ERROR')

from deepface import DeepFace
import cv2
import time
import re
import easyocr
import mysql.connector
import numpy as np
from flask_cors import CORS
import pygame
from ultralytics import YOLO
from paddleocr import PaddleOCR


app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
CORS(app, resources={r"/*": {"origins": "*"}})

path = r"C:\Users\PUSHKAR\module1\og_ig\captured_image.png"   # to store original ID-CARD
no_noise_path = r"C:\Users\PUSHKAR\module1\process_id\no_noise.png" # to store preprocessed ID-CARD
id_face_path = r'C:\Users\PUSHKAR\module1\extract_face\extracted_face.jpg'
real_face_path = r"C:\Users\PUSHKAR\module1\realtime_face\realtime_face.png"
oriented_path = r"C:\Users\PUSHKAR\module1\process_id\oriented_image.png"


# Fire detection paths and configurations
MODEL_PATH_FIRE = r"C:\Users\PUSHKAR\Downloads\firebest.pt"
ALARM_SOUND_PATH = r"C:\Users\PUSHKAR\Downloads\fire-alarm-9677.mp3"
FIRE_IMAGE_PATH = r"C:\Users\PUSHKAR\module1\fire_detection\fire_image.jpg"
FIRE_RESULT_PATH = r"C:\Users\PUSHKAR\module1\fire_detection\result_image.jpg"

ELEC_IMAGE_PATH = r"C:\Users\PUSHKAR\module1\electric_image.jpg"
ELEC_RESULT_PATH = r"C:\Users\PUSHKAR\module1\electric_image_result.jpg"
WATER_IMAGE_PATH = r"C:\Users\PUSHKAR\module1\water_image.jpg"
WATER_RESULT_PATH = r"C:\Users\PUSHKAR\module1\water_image_result.jpg"

pygame.mixer.init()
ocr_engine = PaddleOCR(use_angle_cls=False, lang='en')


def detect_orientation(image_path):
    """Detect if the image is portrait or landscape and rotate if needed"""
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image at {image_path}")
        return None
        
    height, width = image.shape[:2]
    
    # Check orientation
    is_landscape = width > height
    
    # If landscape, check if it needs rotation based on text orientation
    if is_landscape:
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply OCR to detect text orientation
        reader = easyocr.Reader(['en'])
        results = reader.readtext(gray)
        
        # If text is found, check its orientation
        if results:
            # Most ID cards have horizontally aligned text
            # If text boxes are taller than they are wide, image may need rotation
            vertical_text_count = 0
            horizontal_text_count = 0
            
            for (bbox, text, prob) in results:
                # Calculate width and height of text box
                (tl, tr, br, bl) = bbox
                w = max(int(tr[0] - tl[0]), int(br[0] - bl[0]))
                h = max(int(bl[1] - tl[1]), int(br[1] - tr[1]))
                
                if h > w:
                    vertical_text_count += 1
                else:
                    horizontal_text_count += 1
            
            # If more vertical text than horizontal, rotate
            if vertical_text_count > horizontal_text_count:
                # Rotate 90 degrees clockwise
                image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
                print("Rotated image to portrait orientation")
    
    return image
def preprocess(image_path):
    """Preprocess image with orientation handling"""
    # First detect and correct orientation
    try:
        image = detect_orientation(image_path)
        
        # Save the oriented image
        cv2.imwrite(oriented_path, image)
        
        # Apply preprocessing steps
        sharpen_kernel = np.array([[0, -1, 0],
                                [-1, 5,-1],
                                [0, -1, 0]])
        sharpened_image = cv2.filter2D(image, -1, sharpen_kernel)
        gray_image = cv2.cvtColor(sharpened_image, cv2.COLOR_BGR2GRAY)
        _, binary_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
        no_noise_image = cv2.GaussianBlur(binary_image, (5, 5), 0)
        
        # Save preprocessed image
        cv2.imwrite(no_noise_path, image)
        
        return oriented_path
    except Exception as e:
        print(f"Error in preprocess function: {e}")
        # Fallback to using original image
        return image_path

def extract_worker_id(image_path):
    """Extract worker ID with improved handling of orientations"""
    # Load the image
    image = cv2.imread(image_path)
    
    reader = easyocr.Reader(['en'])
    
    # Try multiple orientations if needed
    orientations = [0, 90, 270]  # 0, 90, 270 degrees rotation
    best_result = None
    highest_confidence = 0
    
    for angle in orientations:
        # Rotate image if needed
        if angle != 0:
            h, w = image.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, 
                                  borderMode=cv2.BORDER_REPLICATE)
        else:
            rotated = image
            
        # Run OCR
        result = reader.readtext(rotated)
        
        extracted_text = " ".join([detection[1] for detection in result])
        upper_text = extracted_text.upper()
        
        # Look for worker ID pattern with various formats
        patterns = [
            r"WORKER\s*ID\s*:\s*([A-Z0-9@]+)",
            r"WORKER\s*ID\s*([A-Z0-9@]+)",
            r"ID\s*:\s*([A-Z0-9@]+)"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, upper_text)
            if match:
                confidence = 1.0  # Assign a confidence score
                if confidence > highest_confidence:
                    highest_confidence = confidence
                    best_result = match.group(1)
    
    return best_result

def verify_db(id):
    try:
        conn = mysql.connector.connect(
            host="localhost",
            port = 3306,
            user="root",
            password="",
            database="WorkerDB"
        )

        cursor = conn.cursor()
        query = "SELECT COUNT(*) FROM worker_info WHERE worker_id = %s"
        cursor.execute(query, (id,))  
        result = cursor.fetchone()
        if conn.is_connected():
            cursor.close()
            conn.close()
        if result[0] > 0:
            return True
        else:
            return False
    except mysql.connector.Error as err:
        print(f"Error: {err}")  

def extract_id_face(image_path):
    """Extract face from ID card image with orientation handling"""
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image at {image_path}")
        return False
        
    # Try multiple orientations if needed
    orientations = [0, 90, 270]  # 0, 90, 270 degrees rotation
    face_detected = False
    
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    for angle in orientations:
        # Rotate image if needed
        if angle != 0:
            h, w = image.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, 
                                  borderMode=cv2.BORDER_REPLICATE)
        else:
            rotated = image
            
        # Try Haar Cascade first
        gray = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        if len(faces) > 0:
            # Use the largest face
            largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
            x, y, w, h = largest_face
            face = rotated[y:y+h, x:x+w]
            
            # Apply sharpening for better recognition
            sharpen_kernel = np.array([[0, -1, 0],
                                      [-1, 5,-1],
                                      [0, -1, 0]])
            sharpened_face = cv2.filter2D(face, -1, sharpen_kernel)
            
            # Save the face
            cv2.imwrite(id_face_path, sharpened_face)
            print(f"Face detected and extracted at {angle} degrees rotation")
            face_detected = True
            break
    
    # If Haar cascade fails in all orientations, try DeepFace
    if not face_detected:
        try:
            # Try with the original image first
            detections = DeepFace.extract_faces(image, detector_backend='opencv', enforce_detection=False)
            if len(detections) > 0:
                best_detection = max(detections, key=lambda d: d['confidence'])
                face_region = best_detection['facial_area']
                x, y, w, h = face_region['x'], face_region['y'], face_region['w'], face_region['h']
                face = image[y:y+h, x:x+w]
                
                # Apply sharpening
                sharpen_kernel = np.array([[0, -1, 0],
                                          [-1, 5,-1],
                                          [0, -1, 0]])
                sharpened_face = cv2.filter2D(face, -1, sharpen_kernel)
                
                # Save the face
                cv2.imwrite(id_face_path, sharpened_face)
                print("Face detected using DeepFace")
                face_detected = True
        except Exception as e:
            print(f"DeepFace extraction error: {e}")
    
    # If all automatic detection methods fail, use a fallback approach
    # This assumes ID cards usually have face in a specific region
    if not face_detected:
        h, w = image.shape[:2]
        # Try both portrait and landscape orientations
        if h > w:  # Portrait
            face_region = image[int(h/8):int(h/2), int(w/4):int(3*w/4)]
        else:  # Landscape
            face_region = image[int(h/4):int(3*h/4), int(w/2):int(5*w/6)]
            
        sharpen_kernel = np.array([[0, -1, 0],
                                  [-1, 5,-1],
                                  [0, -1, 0]])
        sharpened_face = cv2.filter2D(face_region, -1, sharpen_kernel)
        cv2.imwrite(id_face_path, sharpened_face)
        print("Used fallback face extraction method")
        face_detected = True
    
    return face_detected
def capture_webcam():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        exit()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image")
            break

        cv2.imshow("Press Space to Capture Image", frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == 32:
            cv2.imwrite(path, frame)
            print("Image captured and saved as 'captured_image.png'")
            break
        
        if key == ord('q'):
            print("Exiting...")
            break

    cap.release()
    cv2.destroyAllWindows()

def capture_realtime_face():
    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    if not cap.isOpened():
        print("Error: Could not open webcam")
        exit()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image")
            break
    
        cv2.imshow("Press Space to Capture Image", frame)
        key = cv2.waitKey(1) & 0xFF
        
        if key == 32:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=5, minSize=(30, 30))
            for (x, y, w, h) in faces:
                face = frame[y:y+h, x:x+w]
                sharpen_kernel = np.array([[0, -1, 0],
                                           [-1, 5,-1],
                                           [0, -1, 0]])
                sharpened_image = cv2.filter2D(face, -1, sharpen_kernel)
            
                cv2.imwrite(real_face_path, sharpened_image)
            break
        
        if key == ord('q'):
            print("Exiting...")
            break

    cap.release()
    cv2.destroyAllWindows()


def compare(path1, path2):
    """Compare two face images with improved error handling"""
    # First verify both images exist and are valid
    if not os.path.exists(path1):
        print(f"Error: Path does not exist: {path1}")
        return False
        
    if not os.path.exists(path2):
        print(f"Error: Path does not exist: {path2}")
        return False
    
    # Try to read both images to verify they're valid
    try:
        img1 = cv2.imread(path1)
        if img1 is None:
            print(f"Error: Could not read image at {path1}")
            return False
            
        img2 = cv2.imread(path2)
        if img2 is None:
            print(f"Error: Could not read image at {path2}")
            return False
            
        # Check if images are too small
        if img1.shape[0] < 20 or img1.shape[1] < 20:
            print(f"Error: Image 1 is too small: {img1.shape}")
            return False
            
        if img2.shape[0] < 20 or img2.shape[1] < 20:
            print(f"Error: Image 2 is too small: {img2.shape}")
            return False
    
        # Try using DeepFace with enforce_detection=False
        try:
            result = DeepFace.verify(path1, path2, threshold=0.5, enforce_detection=False)
            print(f"DeepFace verification result: {result}")
            return result['verified']
        except Exception as e:
            print(f"DeepFace verification error: {e}")
            
            # Fallback: try with different model
            try:
                result = DeepFace.verify(path1, path2, threshold=0.5, model_name="VGG-Face", enforce_detection=False)
                print(f"DeepFace fallback verification result: {result}")
                return result['verified']
            except Exception as e2:
                print(f"DeepFace fallback verification error: {e2}")
                return False
                
    except Exception as e:
        print(f"Error in image processing during comparison: {e}")
        return False
    
@app.route('/verify', methods=['GET', 'POST'])
def verify():
    try:
        if 'id_card' not in request.files or 'face_image' not in request.files:
            return jsonify({
                'success': False,
                'message': 'Missing required images'
            })

        # Save uploaded files
        id_card = request.files['id_card']
        face_image = request.files['face_image']
        
        try:
            id_card.save(path)
            print(f"ID card saved successfully to {path}")
        except Exception as e:
            print(f"Error saving ID card: {e}")
            return jsonify({
                'success': False,
                'message': f'Error saving ID card: {str(e)}'
            })
            
        try:
            face_image.save(real_face_path)
            print(f"Face image saved successfully to {real_face_path}")
        except Exception as e:
            print(f"Error saving face image: {e}")
            return jsonify({
                'success': False,
                'message': f'Error saving face image: {str(e)}'
            })

        # Process the ID card with improved error handling
        try:
            processed_path = preprocess(path)
            print(f"Preprocessed image saved to {processed_path}")
        except Exception as e:
            print(f"Error during preprocessing: {e}")
            return jsonify({
                'success': False,
                'message': f'Error during preprocessing: {str(e)}'
            })
        
        try:
            worker_id = extract_worker_id(processed_path)
            print(f"Extracted Worker ID: {worker_id}")
        except Exception as e:
            print(f"Error extracting worker ID: {e}")
            return jsonify({
                'success': False,
                'message': f'Error extracting worker ID: {str(e)}'
            })
        
        if not worker_id:
            return jsonify({
                'success': False,
                'message': 'Could not extract worker ID'
            })
            
        try:
            id_verified = verify_db(worker_id.strip())
            print(f"Database verification result: {id_verified}")
        except Exception as e:
            print(f"Error verifying worker ID in database: {e}")
            return jsonify({
                'success': False,
                'message': f'Error verifying worker ID: {str(e)}'
            })
        
        if not id_verified:
            return jsonify({
                'success': False,
                'message': 'Worker ID Does Not exist in db'
            })
        
        # Extract face from ID card with improved error handling
        try:
            face_extracted = extract_id_face(processed_path)
            print(f"Face extraction result: {face_extracted}")
        except Exception as e:
            print(f"Error extracting face from ID: {e}")
            return jsonify({
                'success': False,
                'message': f'Error extracting face from ID: {str(e)}'
            })
        
        if not face_extracted:
            return jsonify({
                'success': False,
                'message': 'Could not extract face from ID card'
            })
            
        # Compare the extracted face with the real-time face
        try:
            face_match = compare(id_face_path, real_face_path)
            print(f"Face match result: {face_match}")
        except Exception as e:
            print(f"Error comparing faces: {e}")
            return jsonify({
                'success': False,
                'message': f'Error comparing faces: {str(e)}'
            })
        
        if face_match:
            return jsonify({
                'success': True,
                'message': 'FACE MATCH SUCCESSFUL !!!'
            })
        else:
            return jsonify({
                'success': False,
                'message': 'FACE MATCH UNSUCCESSFUL !!!'
            })
            
    except Exception as e:
        print(f"General error during verification: {e}")
        return jsonify({
            'success': False,
            'message': f'Error during verification: {str(e)}'
        })
    

    #FIRE DETECTION START
    
    # Add the fire detection functions
def play_alarm():
    """Play the alarm sound in a loop."""
    pygame.mixer.music.load(ALARM_SOUND_PATH)
    pygame.mixer.music.play(-1)  # Loop indefinitely

def stop_alarm():
    """Stop the alarm sound."""
    pygame.mixer.music.stop()

def preprocess_fire_image(image):
    """Preprocess image for better fire detection."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    hsv[:,:,2] = clahe.apply(hsv[:,:,2])
    processed = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    blurred = cv2.GaussianBlur(processed, (5,5), 0)
    return blurred

def detect_fire(image):
    """Detect fire in an image and return the result."""
    original = image.copy()
    processed = preprocess_fire_image(image)
    
    model = YOLO(MODEL_PATH_FIRE)
    results = model(processed)
    
    fire_detected = False
    detection_info = []
    
    for result in results:
        if result.boxes is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            
            for box, conf in zip(boxes, confs):
                if conf < 0.5:
                    continue
                
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(original, (x1, y1), (x2, y2), (0, 0, 255), 3)
                cv2.putText(original, f"Fire {conf:.2f}", (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
                fire_detected = True
                detection_info.append({
                    'position': [x1, y1, x2, y2],
                    'confidence': float(conf)
                })
    
    return fire_detected, original, detection_info

# Keep your existing /verify endpoint

@app.route('/detect_fire',  methods=['GET', 'POST'])
def fire_detection_endpoint():
    try:
        if 'image' not in request.files:
            return jsonify({
                'success': False,
                'message': 'Missing image file'
            })

        # Save uploaded image
        image_file = request.files['image']
        image_file.save(FIRE_IMAGE_PATH)
        
        # Process image for fire detection
        image = cv2.imread(FIRE_IMAGE_PATH)
        if image is None:
            return jsonify({
                'success': False,
                'message': 'Failed to read image'
            })
            
        fire_detected, result_img, detection_info = detect_fire(image)
        
        # Save result image
        cv2.imwrite(FIRE_RESULT_PATH, result_img)
        
        # Return result
        return jsonify({
            'success': True,
            'fire_detected': fire_detected,
            'detections': detection_info,
            'message': '🚨 FIRE DETECTED! EMERGENCY!' if fire_detected else '✅ No fire detected',
            'result_image_path': FIRE_RESULT_PATH
        })
            
    except Exception as e:
        print(f"Error during fire detection: {e}")
        return jsonify({
            'success': False,
            'message': f'Error during fire detection: {str(e)}'
        })
@app.route('/detect_fire_video', methods=['POST'])
def fire_detection_video_endpoint():
    try:
        # Check if 'video' field is present in the request
        if 'video' not in request.files:
            return jsonify({
                'success': False,
                'message': 'Missing video file'
            })

        # Save the uploaded video
        video_file = request.files['video']
        video_path = os.path.join('data', 'uploaded_video.mp4')
        video_file.save(video_path)

        # Open the video with OpenCV
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return jsonify({
                'success': False,
                'message': 'Failed to open video'
            })

        fire_detected = False
        frame_count = 0
        
        # Process frames
        while True:
            ret, frame = cap.read()
            if not ret:  # End of video
                break
            frame_count += 1
            
            # Process every 10th frame to optimize performance
            if frame_count % 10 == 0:
                detected, _, _ = detect_fire(frame)  # Use your existing detect_fire function
                if detected:
                    fire_detected = True
                    break  # Stop processing once fire is detected (optional)

        # Clean up
        cap.release()
        os.remove(video_path)  # Optional: remove the temporary video file

        # Return the result
        return jsonify({
            'success': True,
            'fire_detected': fire_detected,
            'message': '🚨 FIRE DETECTED! EMERGENCY!' if fire_detected else '✅ No fire detected in video'
        })

    except Exception as e:
        print(f"Error during video fire detection: {e}")
        return jsonify({
            'success': False,
            'message': f'Error during video fire detection: {str(e)}'
        })
    

#METER DETECTION START


###### ELECTRIC METER DETECTION ######
def process_electric_meter(image, threshold=5000):
    """Process image to detect and read electric meter."""
    original = image.copy()
    MODEL_PATH_ELEC=r"C:\Users\PUSHKAR\Downloads\elec.pt"
    
    model = YOLO(MODEL_PATH_ELEC)
    results = model.predict(image, conf=0.5)
    
    reading = None
    alert = False
    
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        
        for box, conf in zip(boxes, confidences):
            if conf < 0.5:
                continue
            
            x1, y1, x2, y2 = map(int, box)
            cropped = image[y1:y2, x1:x2]
            
            # OCR Processing
            ocr_result = ocr_engine.ocr(cropped, cls=False)
            if ocr_result and ocr_result[0]:
                text = ocr_result[0][0][1][0]
                cleaned = re.sub(r'\D', '', text)
                if cleaned:
                    try:
                        reading = int(cleaned)
                        alert = reading > threshold
                        color = (0, 0, 255) if alert else (0, 255, 0)
                        cv2.putText(original, f"{reading}", (x1, y1-10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                    except ValueError:
                        pass
            
            cv2.rectangle(original, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    return original, reading, alert

@app.route('/detect_electric_meter', methods=['POST'])
def electric_meter_detection_endpoint():
    try:
        if 'image' not in request.files:
            return jsonify({
                'success': False,
                'message': 'Missing image file'
            })
        
        # Get threshold from request parameters (default to 5000)
        threshold = request.form.get('threshold', 7000, type=int)
        
        # Save uploaded image
        image_file = request.files['image']
        image_file.save(ELEC_IMAGE_PATH)
        
        # Process image for electric meter detection
        image = cv2.imread(ELEC_IMAGE_PATH)
        if image is None:
            return jsonify({
                'success': False,
                'message': 'Failed to read image'
            })
            
        result_img, reading, alert = process_electric_meter(image, threshold)
        
        # Save result image
        cv2.imwrite(ELEC_RESULT_PATH, result_img)
        
        # Return result
        return jsonify({
            'success': True,
            'meter_reading': reading,
            'threshold_exceeded': alert,
            'message': '🚨 High Electricity Usage Detected!' if alert else '✅ Normal electricity usage',
            'result_image_path': ELEC_RESULT_PATH
        })
            
    except Exception as e:
        print(f"Error during electric meter detection: {e}")
        return jsonify({
            'success': False,
            'message': f'Error during electric meter detection: {str(e)}'
        })

@app.route('/detect_electric_meter_video', methods=['POST'])
def electric_meter_video_endpoint():
    try:
        if 'video' not in request.files:
            return jsonify({
                'success': False,
                'message': 'Missing video file'
            })
        
        # Get threshold from request parameters (default to 5000)
        threshold = request.form.get('threshold', 5000, type=int)
            
        # Save the uploaded video
        video_file = request.files['video']
        video_path = os.path.join('data', 'uploaded_elec_video.mp4')
        video_file.save(video_path)
        
        # Open the video with OpenCV
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return jsonify({
                'success': False,
                'message': 'Failed to open video'
            })
            
        readings = []
        frame_count = 0
        
        # Process frames
        while True:
            ret, frame = cap.read()
            if not ret:  # End of video
                break
                
            frame_count += 1
            
            # Process every 10th frame to optimize performance
            if frame_count % 10 == 0:
                _, reading, _ = process_electric_meter(frame, threshold)
                if reading is not None:
                    readings.append(reading)
        
        # Clean up
        cap.release()
        os.remove(video_path)  # Optional: remove the temporary video file
        
        # Calculate results
        if readings:
            max_reading = max(readings)
            alert = max_reading > threshold
            
            return jsonify({
                'success': True,
                'readings': readings,
                'max_reading': max_reading,
                'threshold_exceeded': alert,
                'message': f'🚨 High Electricity Usage: {max_reading}!' if alert else f'✅ Normal electricity usage: {max_reading}'
            })
        else:
            return jsonify({
                'success': True,
                'message': 'No valid readings detected in video'
            })
                
    except Exception as e:
        print(f"Error during electric meter video detection: {e}")
        return jsonify({
            'success': False,
            'message': f'Error during electric meter video detection: {str(e)}'
        })

###### WATER METER DETECTION ######
def process_water_meter(image, threshold=1000):
    """Process image to detect and read water meter."""
    original = image.copy()
    MODEL_PATH_WATER = r"C:\Users\PUSHKAR\Downloads\water_model_2.pt"
    model = YOLO(MODEL_PATH_WATER)
    results = model(image)
    
    digits = []
    
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy().astype(int)
        
        # Sort boxes from left to right to read meter correctly
        sorted_indices = np.argsort([box[0] for box in boxes])
        
        for i in sorted_indices:
            box = boxes[i]
            conf = confidences[i]
            cls = classes[i]
            
            if conf < 0.5:
                continue
                
            x1, y1, x2, y2 = map(int, box)
            label = str(cls)
            digits.append(label)
            
            cv2.rectangle(original, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(original,
                       f"{label} ({conf:.2f})",
                       (x1 + 10, y1 + 20),
                       cv2.FONT_HERSHEY_SIMPLEX,
                       0.7,
                       (0, 255, 0),
                       2)
    
    reading = None
    alert = False
    
    if digits:
        reading_str = ''.join(digits)
        try:
            reading = int(reading_str)
            alert = reading > threshold
            # Add overall reading to the image
            color = (0, 0, 255) if alert else (0, 255, 0)
            cv2.putText(original, f"Reading: {reading}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        except ValueError:
            pass
    
    return original, reading, alert

@app.route('/detect_water_meter', methods=['POST'])
def water_meter_detection_endpoint():
    try:
        if 'image' not in request.files:
            return jsonify({
                'success': False,
                'message': 'Missing image file'
            })
        
        # Get threshold from request parameters (default to 1000)
        threshold = request.form.get('threshold', 1000, type=int)
        
        # Save uploaded image
        image_file = request.files['image']
        image_file.save(WATER_IMAGE_PATH)
        
        # Process image for water meter detection
        image = cv2.imread(WATER_IMAGE_PATH)
        if image is None:
            return jsonify({
                'success': False,
                'message': 'Failed to read image'
            })
            
        result_img, reading, alert = process_water_meter(image, threshold)
        
        # Save result image
        cv2.imwrite(WATER_RESULT_PATH, result_img)
        
        # Return result
        return jsonify({
            'success': True,
            'meter_reading': reading,
            'threshold_exceeded': alert,
            'message': '🚨 High Water Usage Detected!' if alert else '✅ Normal water usage',
            'result_image_path': WATER_RESULT_PATH
        })
            
    except Exception as e:
        print(f"Error during water meter detection: {e}")
        return jsonify({
            'success': False,
            'message': f'Error during water meter detection: {str(e)}'
        })

@app.route('/detect_water_meter_video', methods=['POST'])
def water_meter_video_endpoint():
    try:
        if 'video' not in request.files:
            return jsonify({
                'success': False,
                'message': 'Missing video file'
            })
        
        # Get threshold from request parameters (default to 1000)
        threshold = request.form.get('threshold', 1000, type=int)
            
        # Save the uploaded video
        video_file = request.files['video']
        video_path = os.path.join('data', 'uploaded_water_video.mp4')
        video_file.save(video_path)
        
        # Open the video with OpenCV
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return jsonify({
                'success': False,
                'message': 'Failed to open video'
            })
            
        readings = []
        frame_count = 0
        
        # Process frames
        while True:
            ret, frame = cap.read()
            if not ret:  # End of video
                break
                
            frame_count += 1
            
            # Process every 10th frame to optimize performance
            if frame_count % 10 == 0:
                _, reading, _ = process_water_meter(frame, threshold)
                if reading is not None:
                    readings.append(reading)
        
        # Clean up
        cap.release()
        os.remove(video_path)  # Optional: remove the temporary video file
        
        # Calculate results
        if readings:
            max_reading = max(readings)
            alert = max_reading > threshold
            
            return jsonify({
                'success': True,
                'readings': readings,
                'max_reading': max_reading,
                'threshold_exceeded': alert,
                'message': f'🚨 High Water Usage: {max_reading}!' if alert else f'✅ Normal water usage: {max_reading}'
            })
        else:
            return jsonify({
                'success': True,
                'message': 'No valid readings detected in video'
            })
                
    except Exception as e:
        print(f"Error during water meter video detection: {e}")
        return jsonify({
            'success': False,
            'message': f'Error during water meter video detection: {str(e)}'
        })
    
if __name__ == '__main__':
    # Ensure data directory exists
    if not os.path.exists('data'):
        os.makedirs('data')
        
    # Ensure all required directories exist
    for directory in [
        os.path.dirname(path), 
        os.path.dirname(no_noise_path),
        os.path.dirname(id_face_path), 
        os.path.dirname(real_face_path),
        os.path.dirname(FIRE_IMAGE_PATH),
        os.path.dirname(FIRE_RESULT_PATH),
        os.path.dirname(ELEC_IMAGE_PATH),
        os.path.dirname(ELEC_RESULT_PATH),
        os.path.dirname(WATER_IMAGE_PATH),
        os.path.dirname(WATER_RESULT_PATH)
    ]:
        if not os.path.exists(directory):
            os.makedirs(directory)
            
    app.run(host='0.0.0.0', port=5000, debug=True)