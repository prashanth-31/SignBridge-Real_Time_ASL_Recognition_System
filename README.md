# Sign-Language-To-Text-and-Speech-Conversion
![The-26-letters-and-10-digits-of-American-Sign-Language-ASL](https://github.com/user-attachments/assets/912c8aa4-309b-4289-8cb0-dfb920d242ad)

## Overview
This project is designed to convert sign language gestures into text and speech. It utilizes machine learning models to recognize sign language from hand gestures, translate it into text, and then convert the text to speech. This solution aims to bridge the communication gap for individuals with hearing impairments by enabling real-time sign language translation.

### Features
- Sign Language Recognition: Uses computer vision techniques to identify hand gestures from video input.
- Text Conversion: Translates the recognized gestures into corresponding text.
- Speech Conversion: Converts the translated text into speech using a text-to-speech (TTS) model.
- Real-time Processing: Provides real-time recognition and conversion.
  
### Requirements
- Software Requirements
- Python 
- OpenCV
- TensorFlow or PyTorch
- Mediapipe (for gesture detection)
- NumPy
- Pygame (for audio playback)
- gTTS (Google Text-to-Speech)
- Flask

### Hardware Requirements
- A camera for hand gesture capture (Webcam or external camera).

### Usage
- Sign a Gesture: Use the webcam to sign gestures for a recognized language. The system will detect your hand gestures in real-time.
- Text Output: The recognized sign language will be converted into the corresponding text, displayed on the screen.
- Speech Output: The text will then be converted into speech, allowing for auditory feedback.
  
### Model Details
- Gesture Recognition Model: The model used for gesture recognition is based on convolutional neural networks (CNN) or hand landmark detection using the MediaPipe library for accurate and real-time gesture detection.
- Text-to-Speech (TTS): Google’s Text-to-Speech API (gTTS) or any other TTS library can be used to convert the recognized text into speech.

### License
- This project is licensed under the MIT License - see the LICENSE file for details.

