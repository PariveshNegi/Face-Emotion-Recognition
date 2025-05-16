# Face Emotion Recognition
## Tip 
You can change the learning parameater by decreasing the batch size from 128 to 100 and increase the epoches > 100 to get good accuracy but dont fell in trap pf overfitting😉. 
## Overview
Face Emotion Recognition is a real-time emotion detection system that uses deep learning to identify emotions from facial expressions. The system utilizes OpenCV for face detection and a pre-trained neural network model to classify emotions.

## Features
- Real-time face detection using OpenCV
- Emotion classification into seven categories: Angry, Disgust, Fear, Happy, Neutral, Sad, and Surprise
- Live webcam-based emotion recognition
- Pre-trained deep learning model for accurate predictions

## Requirements
To run this project, you need to install the following dependencies:

```bash
pip install numpy opencv-python keras tensorflow
```

## Installation & Setup
1. Clone the repository or download the source code.
2. Ensure the required files (`emotiondetection.json` and `emotiondetection.h5`) are available in the project directory.
3. Run the script using the command:

```bash
python realtime.py
```

## How It Works
1. The script loads the pre-trained deep learning model from `emotiondetection.json` and `emotiondetection.h5`.
2. The webcam captures frames in real-time.
3. Faces are detected using OpenCV’s Haar cascade classifier.
4. The detected face regions are preprocessed and fed into the neural network for emotion classification.
5. The predicted emotion is displayed on the screen in real-time.

## Usage
- The program will automatically start detecting faces and emotions once executed.
- To exit the program, press `q` on your keyboard.

## Notes
- Ensure proper lighting conditions for accurate emotion detection.
- The model expects grayscale images of size 48x48 for proper classification.

## License
This project is open-source and available for personal and educational use.

## Author
[Parivesh Negi]

