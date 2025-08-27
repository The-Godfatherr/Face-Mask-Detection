# Face Mask Detection - Run Instructions

## Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

## Installation Steps

1. **Navigate to the project directory:**
   ```bash
   cd "d:/my codes/Face-Mask-Detection-master"
   ```

2. **Install required dependencies:**
   ```bash
   pip install -r requirements_updated.txt
   ```

## Running the Application

### Method 1: Streamlit Web App (Recommended)
```bash
streamlit run app.py
```
This will start the web application at:
- Local URL: http://localhost:8501
- Network URL: http://192.168.x.x:8501 (your local IP)

### Method 2: Command Line Detection

**For image detection:**
```bash
python detect_mask_image.py --image path/to/your/image.jpg
```

**For video detection:**
```bash
python detect_mask_video.py --video path/to/your/video.mp4
```

**For webcam detection:**
```bash
python detect_mask_video.py
```

## Features Available

1. **Web App (app.py)**:
   - Upload images for mask detection
   - Real-time webcam detection
   - Video file upload and processing
   - User-friendly interface

2. **Command Line Tools**:
   - `detect_mask_image.py` - Process single images
   - `detect_mask_video.py` - Process videos or webcam

## Troubleshooting

1. **If you get TensorFlow compatibility warnings:**
   - These are informational and won't affect functionality
   - The app has been updated to work with newer TensorFlow versions

2. **If webcam doesn't work:**
   - Make sure your webcam is connected and accessible
   - Check if other applications aren't using the webcam

3. **If you get dependency errors:**
   ```bash
   pip install --upgrade -r requirements_updated.txt
   ```

## Model Files
The application uses:
- `mask_detector.keras` - Main trained model
- `face_detector/` - OpenCV face detection models

Both are included in the project and will be loaded automatically.
