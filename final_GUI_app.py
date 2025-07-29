# -*- coding: utf-8 -*-

import cv2
import numpy as np
import tensorflow as tf
from tkinter import *
from PIL import Image, ImageTk
import pyttsx3
import time
import mediapipe as mp

# Load the pretrained model
model = tf.keras.models.load_model("C:/Users/prash/Downloads/asl_cnn_model_finetuned.keras")

# Define label mappings
class_names = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X",
               "Y", "Z", "del", "nothing", "space"]

# Initialize Text-to-Speech engine
engine = pyttsx3.init()

# Initialize global variables
detected_letter = ""
sentence = ""
last_detection_time = 0  # For detection interval

# Custom hand detector class
class handDetector():
    def __init__(self, mode=False, maxHands=1, modComplexity=1, detectionCon=0.9, trackCon=0.9):
        self.mode = mode
        self.maxHands = maxHands
        self.modComplexity = modComplexity
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.modComplexity, self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(img_rgb)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)

        return img

    def findPosition(self, img, handNo=0, draw=True):
        lmList = []
        if self.results.multi_hand_landmarks:
            my_hand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(my_hand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
        return lmList


# Function to preprocess the detected hand
def preprocess_hand(image, bbox):
    xmin, ymin, xmax, ymax = bbox
    hand_img = image[ymin:ymax, xmin:xmax]
    if hand_img.size == 0:
        return None
    hand_img = cv2.resize(hand_img, (64, 64))  # Resize to model input size
    hand_img = cv2.cvtColor(hand_img, cv2.COLOR_BGR2RGB)  # Convert to RGB
    hand_img = hand_img / 255.0  # Normalize
    hand_img = np.expand_dims(hand_img, axis=0)  # Add batch dimension
    return hand_img


# Function to predict the gesture
def predict_gesture(image, bbox):
    hand_img = preprocess_hand(image, bbox)
    if hand_img is None:
        return None
    prediction = model.predict(hand_img, verbose=0)
    confidence = np.max(prediction)
    predicted_label_idx = np.argmax(prediction)
    if confidence > 0.5:  # Set the confidence threshold to 60%
        return class_names[predicted_label_idx]
    return None


# Function to update the detected letter and sentence
def update_text(letter):
    global detected_letter, sentence
    if letter == "space":
        sentence += " "
    elif letter:
        sentence += letter
    detected_letter = letter
    detected_letter_label.config(text=f"DETECTED LETTER: {detected_letter}")
    sentence_label.config(text=f"SENTENCE: {sentence}")


# Function to reset the sentence
def reset_text():
    global detected_letter, sentence
    detected_letter = ""
    sentence = ""
    detected_letter_label.config(text="DETECTED LETTER: ")
    sentence_label.config(text="SENTENCE: ")


# Function to convert the sentence to speech and clear it
def speech_to_text():
    global sentence
    if sentence.strip():
        engine.say(sentence.strip())
        engine.runAndWait()
    reset_text()

# Function to add space to the sentence
def add_space():
    global sentence
    sentence += " "
    sentence_label.config(text=f"SENTENCE: {sentence}")

# Function to delete the last character from the sentence
def delete_last():
    global sentence
    sentence = sentence[:-1]
    sentence_label.config(text=f"SENTENCE: {sentence}")

# Webcam feed and Mediapipe processing
def video_loop():
    global detected_letter, last_detection_time
    ret, frame = cap.read()
    if not ret:
        return
    frame = cv2.flip(frame, 1)

    # Detect hands
    img = detector.findHands(frame, draw=True)
    lm_list = detector.findPosition(frame)

    if lm_list:
        xmin = min([lm[1] for lm in lm_list])
        ymin = min([lm[2] for lm in lm_list])
        xmax = max([lm[1] for lm in lm_list])
        ymax = max([lm[2] for lm in lm_list])

        bbox = (xmin - 20, ymin - 20, xmax + 20, ymax + 20)  # Add padding
        bbox = tuple(max(0, v) for v in bbox)  # Ensure bbox stays in frame

        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)

        if time.time() - last_detection_time > 1:
            letter = predict_gesture(frame, bbox)
            if letter:
                update_text(letter)
                last_detection_time = time.time()

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    imgtk = ImageTk.PhotoImage(image=img)
    video_label.imgtk = imgtk
    video_label.configure(image=imgtk)
    root.after(10, video_loop)


# GUI setup
root = Tk()
root.title("ASL Gesture Recognition")
root.geometry("1920x1080")

# Live webcam feed
video_label = Label(root)
video_label.grid(row=0, column=0, padx=10, pady=10, columnspan=2)

# Display an image on the right side
img = Image.open("C:/Users/prash/Downloads/American-Sign-Language-alphabet.jpg")  # Update with your image path
img = img.resize((800, 500), Image.LANCZOS)
img_display = ImageTk.PhotoImage(img)
image_label = Label(root, image=img_display)
image_label.grid(row=0, column=2, padx=10, pady=10)

# Detected gesture display
detected_letter_label = Label(root, text="DETECTED LETTER: ", font=("Arial", 16))
detected_letter_label.grid(row=1, column=0, columnspan=2, pady=5)

sentence_label = Label(root, text="SENTENCE: ", font=("Arial", 16), wraplength=400, justify="left")
sentence_label.grid(row=2, column=0, columnspan=2, pady=5)

# Buttons for Reset and Speech
space_button = Button(root, text="SPACE", command=add_space, font=("Arial", 14), bg="blue", fg="white", width=10)
space_button.grid(row=4, column=0, pady=20)

delete_button = Button(root, text="DELETE", command=delete_last, font=("Arial", 14), bg="orange", fg="white", width=10)
delete_button.grid(row=4, column=1, pady=20)

reset_button = Button(root, text="RESET", command=reset_text, font=("Arial", 14), bg="red", fg="white", width=10)
reset_button.grid(row=3, column=0, pady=20)

speech_button = Button(root, text="SPEECH", command=speech_to_text, font=("Arial", 14), bg="green", fg="white", width=10)
speech_button.grid(row=3, column=1, pady=20)

# Initialize webcam and hand detector
cap = cv2.VideoCapture(0)
detector = handDetector()

# Start video loop
video_loop()
root.mainloop()

# Release resources on close
cap.release()