#!/usr/bin/env python3
# USAGE
# python test_model.py --model mask_detector.model --test_image path_to_test_image.jpg

import argparse
import cv2
import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array

def test_single_image(image_path, model_path):
    """Test the trained model on a single image"""
    # Load the face mask detector model
    print("[INFO] loading face mask detector model...")
    model = load_model(model_path)
    
    # Load the test image
    image = cv2.imread(image_path)
    if image is None:
        print(f"[ERROR] Could not load image from {image_path}")
        return
    
    # Load face detector
    prototxtPath = os.path.sep.join(["face_detector", "deploy.prototxt"])
    weightsPath = os.path.sep.join(["face_detector", "res10_300x300_ssd_iter_140000.caffemodel"])
    net = cv2.dnn.readNet(prototxtPath, weightsPath)
    
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
    
    faces_detected = 0
    
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            faces_detected += 1
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))
            
            # Extract face ROI
            face = image[startY:endY, startX:endX]
            if face.any():
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face = cv2.resize(face, (224, 224))
                face = img_to_array(face)
                face = preprocess_input(face)
                face = np.expand_dims(face, axis=0)
                
                # Make prediction
                (mask, withoutMask) = model.predict(face)[0]
                label = "Mask" if mask > withoutMask else "No Mask"
                confidence_score = max(mask, withoutMask) * 100
                
                print(f"Face {faces_detected}: {label} ({confidence_score:.2f}%)")
                
                # Draw bounding box and label
                color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
                label_text = f"{label}: {confidence_score:.2f}%"
                cv2.putText(image, label_text, (startX, startY - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
                cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)
    
    if faces_detected == 0:
        print("[INFO] No faces detected in the image")
    else:
        # Save and display the result
        output_path = "test_result.jpg"
        cv2.imwrite(output_path, image)
        print(f"[INFO] Result saved to {output_path}")

if __name__ == "__main__":
    # Construct argument parser
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", type=str, default="mask_detector.model",
                   help="path to trained face mask detector model")
    ap.add_argument("-i", "--test_image", type=str, required=True,
                   help="path to test image")
    args = vars(ap.parse_args())
    
    test_single_image(args["test_image"], args["model"])
