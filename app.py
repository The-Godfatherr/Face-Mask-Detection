import streamlit as st
from PIL import Image, ImageEnhance
import numpy as np
import cv2
import os
import tempfile
from tensorflow import keras 
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

# Setting custom Page Title and Icon with changed layout and sidebar state
st.set_page_config(page_title='Face Mask Detector', page_icon='😷', layout='centered', initial_sidebar_state='expanded')

def local_css(file_name):
    """ Method for reading styles.css and applying necessary changes to HTML"""
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

def mask_image(image_path):
    """Process image for mask detection"""
    # load our serialized face detector model from disk
    print("[INFO] loading face detector model...")
    prototxtPath = os.path.sep.join(["face_detector", "deploy.prototxt"])
    weightsPath = os.path.sep.join(["face_detector",
                                    "res10_300x300_ssd_iter_140000.caffemodel"])
    net = cv2.dnn.readNet(prototxtPath, weightsPath)

    # load the face mask detector model from disk
    print("[INFO] loading face mask detector model...")
    model = load_model("mask_detector.keras")  # Updated to use .keras model

    # load the input image from disk and grab the image spatial dimensions
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    (h, w) = image.shape[:2]

    # construct a blob from the image
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300),
                                 (104.0, 177.0, 123.0))

    # pass the blob through the network and obtain the face detections
    print("[INFO] computing face detections...")
    net.setInput(blob)
    detections = net.forward()

    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with the detection
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the confidence is greater than the minimum confidence
        if confidence > 0.5:
            # compute the (x, y)-coordinates of the bounding box for the object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # ensure the bounding boxes fall within the dimensions of the frame
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            # extract the face ROI, convert it from BGR to RGB channel ordering,
            # resize it to 224x224, and preprocess it
            face = image[startY:endY, startX:endX]
            if face.size == 0:
                continue
                
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)
            face = np.expand_dims(face, axis=0)

            # pass the face through the model to determine if the face has a mask or not
            (mask, withoutMask) = model.predict(face)[0]

            # determine the class label and color we'll use to draw the bounding box and text
            label = "Mask" if mask > withoutMask else "No Mask"
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

            # include the probability in the label
            label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

            # display the label and bounding box rectangle on the output frame
            cv2.putText(image, label, (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)
    
    # Convert to RGB for display
    RGB_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return RGB_img

def process_video(video_file):
    """Process uploaded video for mask detection"""
    # Create a temporary file to save the uploaded video
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(video_file.read())
    tfile.close()
    
    # Load face detector and mask detector models
    prototxtPath = os.path.sep.join(["face_detector", "deploy.prototxt"])
    weightsPath = os.path.sep.join(["face_detector", "res10_300x300_ssd_iter_140000.caffemodel"])
    net = cv2.dnn.readNet(prototxtPath, weightsPath)
    model = load_model("mask_detector.keras")  # Updated to use .keras model
    
    # Open the video file
    cap = cv2.VideoCapture(tfile.name)
    output_frames = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
        net.setInput(blob)
        detections = net.forward()
        
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                (startX, startY) = (max(0, startX), max(0, startY))
                (endX, endY) = (min(w - 1, endX), min(h - 1, endY))
                
                face = frame[startY:endY, startX:endX]
                if face.size > 0:
                    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                    face = cv2.resize(face, (224, 224))
                    face = img_to_array(face)
                    face = preprocess_input(face)
                    face = np.expand_dims(face, axis=0)
                    
                    (mask, withoutMask) = model.predict(face)[0]
                    label = "Mask" if mask > withoutMask else "No Mask"
                    color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
                    label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
                    
                    cv2.putText(frame, label, (startX, startY - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
                    cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
        
        output_frames.append(frame)
    
    cap.release()
    os.unlink(tfile.name)
    return output_frames

def webcam_detection():
    """Webcam face mask detection"""
    st.markdown('<h2 align="center">Detection on Webcam</h2>', unsafe_allow_html=True)
    
    # Load models only once
    prototxtPath = os.path.sep.join(["face_detector", "deploy.prototxt"])
    weightsPath = os.path.sep.join(["face_detector", "res10_300x300_ssd_iter_140000.caffemodel"])
    net = cv2.dnn.readNet(prototxtPath, weightsPath)
    
    # Start webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        st.error("Could not open webcam")
        return
    
    stframe = st.empty()
    stop_button = st.button("Stop Webcam")
    
    # Load model outside the loop to avoid multiple loads
    model = load_model("mask_detector.keras")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture frame from webcam")
            break
            
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
        net.setInput(blob)
        detections = net.forward()
        
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                (startX, startY) = (max(0, startX), max(0, startY))
                (endX, endY) = (min(w - 1, endX), min(h - 1, endY))
                
                face = frame[startY:endY, startX:endX]
                if face.size > 0:
                    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                    face = cv2.resize(face, (224, 224))
                    face = img_to_array(face)
                    face = preprocess_input(face)
                    face = np.expand_dims(face, axis=0)
                    
                    (mask, withoutMask) = model.predict(face, verbose=0)[0]
                    label = "Mask" if mask > withoutMask else "No Mask"
                    color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
                    label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
                    
                    cv2.putText(frame, label, (startX, startY - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
                    cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
        
        # Convert to RGB for display
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        stframe.image(frame_rgb, channels="RGB", use_column_width=True)
        
        # Check if stop button was pressed
        if stop_button:
            break
    
    cap.release()
    # Clear the model from memory to avoid TensorFlow context issues
    del model

def mask_detection():
    print("Mask detection function called")  # Debug statement
    local_css("css/styles.css")
    st.markdown('<h1 align="center">😷 Face Mask Detection</h1>', unsafe_allow_html=True)
    
    activities = ["Image", "Video", "Webcam"]
    st.sidebar.markdown("# Mask Detection on?")
    choice = st.sidebar.selectbox("Choose among the given options:", activities)
    print(f"User selected: {choice}")  # Debug statement

    if choice == 'Image':
        st.markdown('<h2 align="center">Detection on Image</h2>', unsafe_allow_html=True)
        st.markdown("### Upload your image here ⬇")
        image_file = st.file_uploader("Choose an image file", type=['jpg', 'jpeg', 'png'], label_visibility="hidden")
        
        if image_file is not None:
            try:
                our_image = Image.open(image_file)
                # Save the image temporarily
                temp_image_path = "./images/out.jpg"
                os.makedirs(os.path.dirname(temp_image_path), exist_ok=True)
                our_image.save(temp_image_path)
                
                st.image(image_file, caption='Uploaded Image', use_column_width=True)
                st.markdown('<h3 align="center">Image uploaded successfully!</h3>', unsafe_allow_html=True)
                
                if st.button('Process'):
                    with st.spinner('Processing image...'):
                        processed_image = mask_image(temp_image_path)
                        st.image(processed_image, caption='Processed Image', use_column_width=True)
                        st.success('Image processing completed!')
                        
            except Exception as e:
                st.error(f"Error processing image: {e}")

    elif choice == 'Video':
        st.markdown('<h2 align="center">Detection on Video</h2>', unsafe_allow_html=True)
        st.markdown("### Upload your video here ⬇")
        video_file = st.file_uploader("Choose a video file", type=['mp4', 'avi', 'mov'], label_visibility="hidden")
        
        if video_file is not None:
            try:
                st.markdown('<h3 align="center">Video uploaded successfully!</h3>', unsafe_allow_html=True)
                
                if st.button('Process Video'):
                    with st.spinner('Processing video...'):
                        processed_frames = process_video(video_file)
                        if processed_frames:
                            st.success('Video processing completed!')
                            # Display first processed frame as preview
                            st.image(processed_frames[0], caption='First processed frame', use_column_width=True)
                            
                            # Option to download processed video
                            st.info("Video processing complete. The processed frames are available.")
                        else:
                            st.error('No faces detected in the video')
            except Exception as e:
                st.error(f"Error processing video: {e}")

    elif choice == 'Webcam':
        webcam_detection()

# Run the application
if __name__ == "__main__":
    mask_detection()
