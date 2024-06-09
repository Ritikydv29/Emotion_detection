import tkinter as tk
from tkinter import filedialog, Label, Button
from tensorflow.keras.models import model_from_json
from PIL import Image, ImageTk
import numpy as np
import cv2

# Function to load the model
def FacialExpressionModel(json_file, weights_file):
    with open(json_file, "r") as file:
        loaded_model_json = file.read()
        model = model_from_json(loaded_model_json)
    model.load_weights(weights_file)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Function to detect emotion
def Detect(file_path):
    global label1, sign_image
    
    # Read the image
    image = cv2.imread(file_path)
    if image is None:
        label1.configure(foreground="#011638", text="Unable to read image")
        return
    
    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = facec.detectMultiScale(gray_image, scaleFactor=1.02, minNeighbors=7, minSize=(30, 30))
    
    if len(faces) == 0:
        label1.configure(foreground="#011638", text="No face detected")
        return

    for (x, y, w, h) in faces:
        # Draw a rectangle around the faces
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        # Preprocess the face region for prediction
        fc = gray_image[y:y+h, x:x+w]
        roi = cv2.resize(fc, (48, 48))
        roi = np.expand_dims(roi, axis=0)
        roi = roi / 255.0  # Normalize
        
        # Predict emotion
        pred = EMOTIONS_LIST[np.argmax(model.predict(roi))]
        print("Predicted Emotion: " + pred)
        label1.configure(foreground="#011638", text=pred)
        
        # Show the image with rectangles
        img_with_faces = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_with_faces = Image.fromarray(img_with_faces)
        img_with_faces = ImageTk.PhotoImage(img_with_faces)
        
        sign_image.configure(image=img_with_faces)
        sign_image.image = img_with_faces

# Function to show the Detect button
def show_Detect_button(file_path):
    global detect_b
    detect_b = Button(top, text="Detect Emotion", command=lambda: Detect(file_path), padx=10, pady=5)
    detect_b.configure(background="#364156", foreground='white', font=('arial', 10, 'bold'))
    detect_b.place(relx=0.79, rely=0.46)

# Function to upload image
def upload_image():
    try:
        file_path = filedialog.askopenfilename()
        if not file_path:
            return
        uploaded = Image.open(file_path)
        uploaded.thumbnail((int(top.winfo_width()/2.3), int(top.winfo_height()/2.3)))
        im = ImageTk.PhotoImage(uploaded)
        sign_image.configure(image=im)
        sign_image.image = im
        label1.configure(text='')
        show_Detect_button(file_path)
    except Exception as e:
        print(e)

# Main GUI setup
top = tk.Tk()
top.geometry('800x600')
top.title('Emotion Detector')
top.configure(background='#CDCDCD')

label1 = Label(top, background='#CDCDCD', font=('arial', 15, 'bold'))
sign_image = Label(top)

# Load the face detection model
facec = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Load the emotion detection model
model = FacialExpressionModel("model_a.json", "model.weights.h5")

# List of emotions
EMOTIONS_LIST = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

# Upload button
upload = Button(top, text="Upload Image", command=upload_image, padx=10, pady=5)
upload.configure(background="#364156", foreground='white', font=('arial', 10, 'bold'))
upload.pack(side='bottom', pady=50)

sign_image.pack(side='bottom', expand=True)
label1.pack(side='bottom', expand=True)

heading = Label(top, text='Emotion Detector', pady=20, font=('arial', 25, 'bold'))
heading.configure(background='#CDCDCD', foreground="#364156")
heading.pack()

top.mainloop()
