from flask import Flask, request, jsonify, send_file, after_this_request

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
tf.get_logger().setLevel('ERROR')
import supervision as sv
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
from inference_sdk import InferenceHTTPClient
import base64
import threading
import time
from flask_socketio import SocketIO, emit
import io
import tempfile

app = Flask(__name__)
socketio = SocketIO(app)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
CORS(app, resources={r"/*": {"origins": "*"}})

path = r"C:\Users\PUSHKAR\module1\og_ig\captured_image.png"   # to store original ID-CARD
no_noise_path = r"C:\Users\PUSHKAR\module1\process_id\no_noise.png" # to store preprocessed ID-CARD
id_face_path = r'C:\Users\PUSHKAR\module1\extract_face\extracted_face.jpg'
real_face_path = r"C:\Users\PUSHKAR\module1\realtime_face\realtime_face.png"
oriented_path = r"C:\Users\PUSHKAR\module1\process_id\oriented_image.png"
socketio = SocketIO(app, cors_allowed_origins="*")


# Fire detection paths and configurations
MODEL_PATH_FIRE = r"C:\Users\PUSHKAR\Downloads\firebest.pt"
ALARM_SOUND_PATH = r"C:\Users\PUSHKAR\Downloads\fire-alarm-9677.mp3"
FIRE_IMAGE_PATH = r"C:\Users\PUSHKAR\module1\fire_detection\fire_image.jpg"
FIRE_RESULT_PATH = r"C:\Users\PUSHKAR\module1\fire_detection\result_image.jpg"

ELEC_IMAGE_PATH = r"C:\Users\PUSHKAR\module1\electric\electric_image.jpg"
ELEC_RESULT_PATH = r"C:\Users\PUSHKAR\module1\electric\electric_image_result.jpg"
WATER_IMAGE_PATH = r"C:\Users\PUSHKAR\module1\water\water_image.jpg"
WATER_RESULT_PATH = r"C:\Users\PUSHKAR\module1\water\water_image_result.jpg"

# Safety detection paths and configurations
SAFETY_IMAGE_PATH = r"C:\Users\PUSHKAR\module1\safety_detection\safety_image.jpg"
SAFETY_RESULT_PATH = r"C:\Users\PUSHKAR\module1\safety_detection\safety_result.jpg"
SAFETY_VIDEO_PATH = r"C:\Users\PUSHKAR\module1\safety_detection\safety_video.mp4"
SAFETY_VIDEO_RESULT_PATH = r"C:\Users\PUSHKAR\module1\safety_detection\safety_video_result.mp4"

TRUCK_MODEL_PATH = r"G:\Module4Images\construction_version_2_Feb\weights\best.pt"
TRUCK_IMAGE_PATH = r"C:\Users\PUSHKAR\module1\truck\truck_image.jpg"
TRUCK_RESULT_PATH = r"C:\Users\PUSHKAR\module1\truck\truck_image_result.jpg"
TRUCK_VIDEO_PATH = r"C:\Users\PUSHKAR\module1\truck\truck_video.jpg"
TRUCK_VIDEO_RESULT_PATH = r"C:\Users\PUSHKAR\module1\truck\truck_video_result.jpg"

# Define the paths for vehicle detection
VEHICLE_MODEL_PATH = r"C:\Users\PUSHKAR\Downloads\best.pt"
VEHICLE_IMAGE_PATH = r"C:\Users\PUSHKAR\module1\vehicle\vehicle_image.jpg"
VEHICLE_RESULT_PATH = r"C:\Users\PUSHKAR\module1\vehicle\vehicle_image_result.jpg"
VEHICLE_VIDEO_PATH = r"C:\Users\PUSHKAR\module1\vehicle\vehicle_video.jpg"
VEHICLE_VIDEO_RESULT_PATH = r"C:\Users\PUSHKAR\module1\vehicle\vehicle_video_result.jpg"
vehicle_model = YOLO(VEHICLE_MODEL_PATH)
class_names = vehicle_model.names

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
def process_electric_meter(image, threshold):
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
            print(reading)
    
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
        threshold = request.form.get('threshold', 2000, type=int)
        
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
    
#SAFETY GEAR DETECTION
# Dictionary to store active stream sessions
active_streams = {}

def process_safety_detection(image, confidence_threshold=0.5):
    """Process image to detect safety gear."""
    original = image.copy()
    SAFETY_MODEL_PATH=r"C:\Users\PUSHKAR\Downloads\best_security.pt"
    model = YOLO(SAFETY_MODEL_PATH)
    results = model(image)
    
    detections = []
    
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy().astype(int)
        
        for i, box in enumerate(boxes):
            conf = confidences[i]
            cls = classes[i]
            
            if conf < confidence_threshold:
                continue
                
            x1, y1, x2, y2 = map(int, box)
            label = model.names[cls]
            
            detections.append({
                "label": label,
                "confidence": float(conf),
                "bbox": [int(x1), int(y1), int(x2), int(y2)]
            })
            
            cv2.rectangle(original, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(original,
                       f"{label} ({conf:.2f})",
                       (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX,
                       0.5,
                       (0, 255, 0),
                       2)
    
    return original, detections

@app.route('/detect_safety_gear', methods=['POST'])
def safety_gear_detection_endpoint():
    try:
        if 'image' not in request.files:
            return jsonify({
                'success': False,
                'message': 'Missing image file'
            })
        
        # Get confidence threshold from request parameters (default to 0.5)
        confidence = request.form.get('confidence', 0.5, type=float)
        
        # Save uploaded image
        image_file = request.files['image']
        image_file.save(SAFETY_IMAGE_PATH)
        
        # Process image for safety gear detection
        image = cv2.imread(SAFETY_IMAGE_PATH)
        if image is None:
            return jsonify({
                'success': False,
                'message': 'Failed to read image'
            })
            
        result_img, detections = process_safety_detection(image, confidence)
        
        # Save result image
        cv2.imwrite(SAFETY_RESULT_PATH, result_img)
        
        # Create summary of detections
        detection_summary = {}
        for det in detections:
            label = det["label"]
            if label in detection_summary:
                detection_summary[label] += 1
            else:
                detection_summary[label] = 1
        
        # Encode the result image to base64 to send back to client
        _, buffer = cv2.imencode('.jpg', result_img)
        result_image_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # Return result
        return jsonify({
            'success': True,
            'detections': detections,
            'detection_summary': detection_summary,
            'result_image_base64': result_image_base64,
            'result_image_path': SAFETY_RESULT_PATH
        })
            
    except Exception as e:
        print(f"Error during safety gear detection: {e}")
        return jsonify({
            'success': False,
            'message': f'Error during safety gear detection: {str(e)}'
        })

@app.route('/detect_safety_gear_video', methods=['POST'])
def safety_gear_video_endpoint():
    try:
        if 'video' not in request.files:
            return jsonify({
                'success': False,
                'message': 'Missing video file'
            })
        
        # Get confidence threshold from request parameters
        confidence = request.form.get('confidence', 0.5, type=float)
        
        # Save the uploaded video
        video_file = request.files['video']
        video_path = SAFETY_VIDEO_PATH
        video_file.save(video_path)
        
        # Open the video with OpenCV
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return jsonify({
                'success': False,
                'message': 'Failed to open video'
            })
        
        # Get video properties    
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Initialize video writer for result
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(SAFETY_VIDEO_RESULT_PATH, fourcc, fps, (width, height))
            
        all_detections = []
        frame_count = 0
        detection_summary = {}
        
        # Process frames
        while True:
            ret, frame = cap.read()
            if not ret:  # End of video
                break
                
            frame_count += 1
            
            # Process every 5th frame to optimize performance
            if frame_count % 5 == 0:
                result_frame, detections = process_safety_detection(frame, confidence)
                
                # Update detection summary
                for det in detections:
                    label = det["label"]
                    if label in detection_summary:
                        detection_summary[label] += 1
                    else:
                        detection_summary[label] = 1
                
                # Save frame detections
                frame_data = {
                    "frame": frame_count,
                    "detections": detections
                }
                all_detections.append(frame_data)
                
                # Write the processed frame
                out.write(result_frame)
            else:
                # Write original frame for skipped frames
                out.write(frame)
        
        # Clean up
        cap.release()
        out.release()
        
        # Return results
        return jsonify({
            'success': True,
            'frames_processed': frame_count,
            'detection_summary': detection_summary,
            'detections': all_detections,
            'result_video_path': SAFETY_VIDEO_RESULT_PATH
        })
                
    except Exception as e:
        print(f"Error during safety gear video detection: {e}")
        return jsonify({
            'success': False,
            'message': f'Error during safety gear video detection: {str(e)}'
        })

# Socket.IO event handlers for real-time streaming
@socketio.on('connect')
def handle_connect():
    print(f"Client connected: {request.sid}")

@socketio.on('disconnect')
def handle_disconnect():
    if request.sid in active_streams:
        active_streams[request.sid]['active'] = False
        del active_streams[request.sid]
    print(f"Client disconnected: {request.sid}")

@socketio.on('start_stream')
def handle_start_stream(data):
    print(f"Starting stream for client: {request.sid}")
    
    # Store configuration in active streams
    active_streams[request.sid] = {
        'active': True,
        'confidence': data.get('confidence', 0.5)
    }
    
    # Send acknowledgment
    emit('stream_started', {'status': 'Stream started'})

@socketio.on('stop_stream')
def handle_stop_stream():
    if request.sid in active_streams:
        active_streams[request.sid]['active'] = False
        emit('stream_stopped', {'status': 'Stream stopped'})

@socketio.on('frame')
def handle_frame(data):
    if request.sid not in active_streams or not active_streams[request.sid]['active']:
        return
    
    try:
        # Decode the image from base64
        image_data = base64.b64decode(data['image'])
        nparr = np.frombuffer(image_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Process the frame
        confidence = active_streams[request.sid]['confidence']
        result_frame, detections = process_safety_detection(frame, confidence)
        
        # Create detection summary
        detection_summary = {}
        for det in detections:
            label = det["label"]
            if label in detection_summary:
                detection_summary[label] += 1
            else:
                detection_summary[label] = 1
        
        # Encode the processed frame to base64
        _, buffer = cv2.imencode('.jpg', result_frame)
        result_image_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # Send the processed frame back to the client
        emit('processed_frame', {
            'image': result_image_base64,
            'detections': detections,
            'detection_summary': detection_summary
        })
        
    except Exception as e:
        print(f"Error processing frame: {e}")
        emit('error', {'message': str(e)})

#TRuCK DETECTION STARTS HERE

# Truck detection functions
def process_truck_frame(frame):
    """Process a single frame to detect trucks."""
    # Run inference
    truck_model = YOLO(TRUCK_MODEL_PATH)
    class_names = truck_model.names
    vehicle_results = truck_model.predict(frame)

    # Process vehicle model results
    vehicle_boxes = []
    vehicle_confidences = []
    vehicle_class_ids = []
    for result in vehicle_results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = box.conf[0].item()
            class_id = int(box.cls[0])
            
            # Ensure boxes stay within image boundaries
            orig_width, orig_height = frame.shape[1], frame.shape[0]
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(orig_width - 1, x2), min(orig_height - 1, y2)

            vehicle_boxes.append([x1, y1, x2, y2])
            vehicle_confidences.append(confidence)
            vehicle_class_ids.append(class_id)

    if vehicle_boxes:
        vehicle_detections = sv.Detections(
            xyxy=np.array(vehicle_boxes, dtype=np.float32),
            confidence=np.array(vehicle_confidences, dtype=np.float32),
            class_id=np.array(vehicle_class_ids, dtype=np.int64)
        )
        combined_detections = vehicle_detections.with_nms(threshold=0.45, class_agnostic=False)
    else:
        combined_detections = sv.Detections(
            xyxy=np.array([], dtype=np.float32).reshape(0, 4),
            confidence=np.array([], dtype=np.float32),
            class_id=np.array([], dtype=np.int64)
        )

    # Draw bounding boxes
    processed_frame = frame.copy()
    detections = []
    
    if len(combined_detections.xyxy) > 0:
        for box, confidence, class_id in zip(combined_detections.xyxy.tolist(), combined_detections.confidence.tolist(), combined_detections.class_id.tolist()):
            x1, y1, x2, y2 = map(int, box)
            color = (255, 0, 0)  # Red for vehicle classes
            
            # Get label
            label = f"{class_names[class_id]}"
            
            # Add to detections list
            detections.append({
                'class': label,
                'confidence': round(confidence, 2),
                'box': [x1, y1, x2, y2]
            })
            
            # Draw rectangle and label inside bounding box
            cv2.rectangle(processed_frame, (x1, y1), (x2, y2), color, 2)
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            text_x = x1 + (x2 - x1 - label_size[0]) // 2
            text_y = y1 + (y2 - y1 + label_size[1]) // 2
            
            cv2.putText(processed_frame, label, (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    return processed_frame, detections

# Endpoints for truck detection
@app.route('/detect_truck', methods=['POST'])
def truck_detection_endpoint():
    """Endpoint to detect trucks in a single image."""
    try:
        if 'image' not in request.files:
            return jsonify({
                'success': False,
                'message': 'Missing image file'
            })
        
        # Save uploaded image
        image_file = request.files['image']
        image_file.save(TRUCK_IMAGE_PATH)
        
        # Process image for truck detection
        image = cv2.imread(TRUCK_IMAGE_PATH)
        if image is None:
            return jsonify({
                'success': False,
                'message': 'Failed to read image'
            })
            
        result_img, detections = process_truck_frame(image)
        
        # Save result image
        cv2.imwrite(TRUCK_RESULT_PATH, result_img)
        
        # Return result
        return jsonify({
            'success': True,
            'detections': detections,
            'detection_count': len(detections),
            'message': f'Detected {len(detections)} vehicles/trucks',
            'result_image_path': TRUCK_RESULT_PATH
        })
            
    except Exception as e:
        print(f"Error during truck detection: {e}")
        return jsonify({
            'success': False,
            'message': f'Error during truck detection: {str(e)}'
        })

@app.route('/detect_truck_video', methods=['POST'])
def truck_video_endpoint():
    """Endpoint to detect trucks in a video."""
    try:
        if 'video' not in request.files:
            return jsonify({
                'success': False,
                'message': 'Missing video file'
            })
            
        # Save the uploaded video
        video_file = request.files['video']
        video_file.save(TRUCK_VIDEO_PATH)
        
        # Open the video with OpenCV
        cap = cv2.VideoCapture(TRUCK_VIDEO_PATH)
        if not cap.isOpened():
            return jsonify({
                'success': False,
                'message': 'Failed to open video'
            })
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Set up VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
        out = cv2.VideoWriter(TRUCK_VIDEO_RESULT_PATH, fourcc, fps, (width, height))
        
        # Process frames
        frame_count = 0
        all_detections = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            
            # Process every 10th frame to optimize performance
            if frame_count % 10 == 0:
                processed_frame, frame_detections = process_truck_frame(frame)
                out.write(processed_frame)
                all_detections.extend(frame_detections)
            else:
                out.write(frame)
        
        # Clean up
        cap.release()
        out.release()
        
        # Calculate results
        detection_counts = {}
        for detection in all_detections:
            class_name = detection['class']
            if class_name in detection_counts:
                detection_counts[class_name] += 1
            else:
                detection_counts[class_name] = 1
        
        return jsonify({
            'success': True,
            'total_detections': len(all_detections),
            'detection_counts': detection_counts,
            'processed_frames': frame_count // 10,
            'result_video_path': TRUCK_VIDEO_RESULT_PATH,
            'message': f'Processed video with {len(all_detections)} total detections'
        })
                
    except Exception as e:
        print(f"Error during truck video detection: {e}")
        return jsonify({
            'success': False,
            'message': f'Error during truck video detection: {str(e)}'
        })
    
# Vehicle detection functions
def process_vehicle_frame(frame):
    """Process a single frame to detect vehicles."""
    # Run inference
    
    vehicle_results = vehicle_model.predict(frame)

    # Process vehicle model results
    vehicle_boxes = []
    vehicle_confidences = []
    vehicle_class_ids = []
    for result in vehicle_results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = box.conf[0].item()
            class_id = int(box.cls[0])
            
            # Ensure boxes stay within image boundaries
            orig_width, orig_height = frame.shape[1], frame.shape[0]
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(orig_width - 1, x2), min(orig_height - 1, y2)

            vehicle_boxes.append([x1, y1, x2, y2])
            vehicle_confidences.append(confidence)
            vehicle_class_ids.append(class_id)

    if vehicle_boxes:
        vehicle_detections = sv.Detections(
            xyxy=np.array(vehicle_boxes, dtype=np.float32),
            confidence=np.array(vehicle_confidences, dtype=np.float32),
            class_id=np.array(vehicle_class_ids, dtype=np.int64)
        )
        combined_detections = vehicle_detections.with_nms(threshold=0.45, class_agnostic=False)
    else:
        combined_detections = sv.Detections(
            xyxy=np.array([], dtype=np.float32).reshape(0, 4),
            confidence=np.array([], dtype=np.float32),
            class_id=np.array([], dtype=np.int64)
        )

    # Draw bounding boxes
    processed_frame = frame.copy()
    detections = []
    
    if len(combined_detections.xyxy) > 0:
        for box, confidence, class_id in zip(combined_detections.xyxy.tolist(), combined_detections.confidence.tolist(), combined_detections.class_id.tolist()):
            x1, y1, x2, y2 = map(int, box)
            
            # Different color for different vehicle types
            if class_names[class_id] == "car":
                color = (0, 255, 0)  # Green for cars
            elif class_names[class_id] == "motorcycle":
                color = (0, 0, 255)  # Blue for motorcycles
            elif class_names[class_id] == "bus":
                color = (255, 0, 255)  # Purple for buses
            else:
                color = (0, 255, 255)  # Yellow for other vehicles
            
            # Get label
            label = f"{class_names[class_id]}"
            
            # Add to detections list
            detections.append({
                'class': label,
                'confidence': round(confidence, 2),
                'box': [x1, y1, x2, y2]
            })
            
            # Draw rectangle and label inside bounding box
            cv2.rectangle(processed_frame, (x1, y1), (x2, y2), color, 2)
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            text_x = x1 + (x2 - x1 - label_size[0]) // 2
            text_y = y1 + (y2 - y1 + label_size[1]) // 2
            
            cv2.putText(processed_frame, label, (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    return processed_frame, detections

# Endpoints for vehicle detection
@app.route('/detect_vehicle', methods=['POST'])
def vehicle_detection_endpoint():
    """Endpoint to detect vehicles in a single image and return the processed image."""
    try:
        # Check if an image file is provided in the request
        if 'image' not in request.files:
            return jsonify({
                'success': False,
                'message': 'Missing image file'
            }), 400
        
        # Read the uploaded image file
        image_file = request.files['image']
        image = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({
                'success': False,
                'message': 'Failed to read image'
            }), 400
            
        # Process the image to detect vehicles and draw bounding boxes
        result_img, _ = process_vehicle_frame(image)
        
        # Encode the processed image to JPEG in memory
        _, buffer = cv2.imencode('.jpg', result_img)
        
        # Send the processed image as a response
        return send_file(
            io.BytesIO(buffer),
            mimetype='image/jpeg',
            as_attachment=True,
            download_name='vehicle_detection_result.jpg'
        )
        
    except Exception as e:
        print(f"Error during vehicle detection: {e}")
        return jsonify({
            'success': False,
            'message': f'Error during vehicle detection: {str(e)}'
        }), 500
# Fix for the video output issue
@app.route('/detect_vehicle_video', methods=['POST'])
def vehicle_video_endpoint():
    """Endpoint to detect vehicles in a video and return the processed video."""
    try:
        # Check if a video file is provided in the request
        if 'video' not in request.files:
            return jsonify({
                'success': False,
                'message': 'Missing video file'
            }), 400
            
        # Save the uploaded video to a temporary file
        video_file = request.files['video']
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_video:
            video_file.save(temp_video.name)
            temp_video_path = temp_video.name
        
        # Open the video for processing
        cap = cv2.VideoCapture(temp_video_path)
        if not cap.isOpened():
            os.remove(temp_video_path)
            return jsonify({
                'success': False,
                'message': 'Failed to open video'
            }), 400
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Create a temporary file for the output video
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_result:
            temp_result_path = temp_result.name
        
        # Set up VideoWriter with the MP4 codec
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_result_path, fourcc, fps, (width, height))
        
        if not out.isOpened():
            cap.release()
            os.remove(temp_video_path)
            return jsonify({
                'success': False,
                'message': 'Failed to create output video file'
            }), 500
        
        # Process video frames
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            # Process every 5th frame for performance (adjust as needed)
            if frame_count % 5 == 0:
                processed_frame, _ = process_vehicle_frame(frame)
                out.write(processed_frame)
            else:
                out.write(frame)
        
        # Release resources
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        
        # Register a cleanup function to delete temporary files after the response
        @after_this_request
        def delete_temp_files(response):
            try:
                os.remove(temp_video_path)
                os.remove(temp_result_path)
            except Exception as e:
                print(f"Error deleting temporary files: {e}")
            return response
        
        # Send the processed video as a response
        return send_file(
            temp_result_path,
            mimetype='video/mp4',
            as_attachment=True,
            download_name='vehicle_detection_result.mp4'
        )
        
    except Exception as e:
        print(f"Error during vehicle video detection: {e}")
        return jsonify({
            'success': False,
            'message': f'Error during vehicle video detection: {str(e)}'
        }), 500   
    
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
        os.path.dirname(WATER_RESULT_PATH),
        os.path.dirname(SAFETY_IMAGE_PATH),
        os.path.dirname(SAFETY_RESULT_PATH),
        os.path.dirname(SAFETY_VIDEO_PATH),
        os.path.dirname(SAFETY_VIDEO_RESULT_PATH),
        os.path.dirname(TRUCK_IMAGE_PATH),
        os.path.dirname(TRUCK_RESULT_PATH),
        os.path.dirname(TRUCK_VIDEO_PATH),
        os.path.dirname(TRUCK_VIDEO_RESULT_PATH),
        os.path.dirname(VEHICLE_IMAGE_PATH),
        os.path.dirname(VEHICLE_RESULT_PATH),
        os.path.dirname(VEHICLE_VIDEO_PATH),
        os.path.dirname(VEHICLE_VIDEO_RESULT_PATH),
        
    ]:
        if not os.path.exists(directory):
            os.makedirs(directory)
            
    app.run(host='0.0.0.0', port=5000, debug=True)