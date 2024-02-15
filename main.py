import cv2
import numpy as np
from keras.models import load_model
from matplotlib import pyplot
from ultralytics import YOLO
import pygame
import threading
import pickle
import math
import cvzone
import nltk
import tkinter as tk
import sqlite3

nltk.download('punkt')

model2 = load_model('model.h5')
model3= load_model('model_saved3.h5')
model4= load_model('mode.h5')
model6=load_model('mode55.h5')
model_yawn=load_model('drowsy5.h5')
model_mask=load_model('Masks2.h5')
model5 = YOLO("../Yolo-Weights/yolov8l.pt")

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

# Initialize pygame mixer
pygame.mixer.init()

# Load the saved model
with open('model_saved2 (2)', 'rb') as file:
    model_feed = pickle.load(file)

# Load the saved vectorizer
with open('model_vector3', 'rb') as file:
    bow_counts = pickle.load(file)  # Load the saved vectorizer


# Function to save task to the database
def save_to_database(new_task):
    # Connect to the SQLite database (create one if it doesn't exist)
    conn = sqlite3.connect('feedback_database.db')
    c = conn.cursor()

    # Create table if not exists
    c.execute('''CREATE TABLE IF NOT EXISTS Feedbacks
                 (Name TEXT, Email TEXT, Phone TEXT, Feedback TEXT)''')

    # Insert the task into the database
    c.execute("INSERT INTO Feedbacks (Name, Email, Phone, Feedback) VALUES (?, ?, ?, ?)",
              ('John Doe', 'john@example.com', '1234567890', new_task))

    # Commit changes and close connection
    conn.commit()
    conn.close()


#Virtual Assistant feedback
def single_response(new_review):
    # Fit the new review to the existing vocabulary before transforming
    new_review_bow = bow_counts.transform([new_review.lower()])  # Transform the new review
    predicted_sentiment = model_feed.predict(new_review_bow)  # Make a prediction
    print("Predicted Sentiment:", predicted_sentiment)
    if predicted_sentiment[0] == "Positive":
        print("Thank you for your trust, we appreciate that you are glad with our service!")

    elif predicted_sentiment[0] == "Negative":
        print("We are sorry to hear that, we are taking your feedback into considerations for futurework.")
    return  predicted_sentiment[0]

# Function to play the alarm sound
def play_alarm_sound():
    pygame.mixer.music.load('response3.mp3')
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)

def play_alarm():
    pygame.mixer.music.load('response4.mp3')
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)

def play_greetings(gender):
    if gender==0:
        pygame.mixer.music.load('resposne2.mp3')
    else:
        pygame.mixer.music.load('response.mp3')
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)

def objectdetection():
    cap = cv2.VideoCapture(0)
    while True:
        success, img = cap.read()
        results = model5(img, stream=True)
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                w, h = x2 - x1, y2 - y1
                cvzone.cornerRect(img, (x1, y1, w, h))
                conf = math.ceil((box.conf[0] * 100)) / 100
                cls = int(box.cls[0])
                cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)), scale=1, thickness=2)

        cv2.imshow("video", img)
        if cv2.waitKey(1) & 0xff == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

mapper5 = {
    0: "Young",
    1: "Middle",
    2: 'Old',
}

mapper3 = {
    0: 'Female',
    1: 'Male'
}

mapper2 = {
    0: 'happy',
    1: 'sad',
    2: 'neutral'
}

mapper1 = {
    0: "angry",
    1: "happy",
    2: "sad",
    3: "surprised",
    4: "neutral",
}

model = load_model('drowiness_new5.h5')

def image_analysis(img):
    # Load the original image
    original_image = cv2.imread(img)

    # Display the original image
    fig, axes = pyplot.subplots(1, 2, figsize=(8, 4), dpi=100)
    axes[0].imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    # Load and process the image for analysis
    input_image = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
    input_image = cv2.resize(input_image, (48, 48))
    input_image = input_image.astype('float32') / 255.0
    input_image = input_image.reshape(48, 48, 1)

    # Make predictions for the input image
    prediction = model4.predict(input_image.reshape(1, 48, 48, 1))[0]
    predicted = np.argmax(prediction)

    # Display the processed image with prediction
    axes[1].imshow(input_image[:, :, 0], cmap='gray')
    axes[1].set_title(f'Prediction: {mapper3[predicted]}')
    axes[1].axis('off')

    pyplot.show()

    print(f"Original Image: {img}")
    print(f"Prediction: {mapper3[predicted]}")

t=0
def perform_video_analysis():
    global t
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    mouth_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
    eye_glasses_cascade= cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcasacde_eye_tree_eyeglasses.xml')
    left_eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_righteye_2splits.xml')
    right_eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_lefteye_2splits.xml')

    capture = cv2.VideoCapture(0)

    while True:
        ret, frame = capture.read()



        if not ret:
            break

        # Resize the frame to a wider size (e.g., double the width)
        new_width = frame.shape[1] * 2-200
        new_height = frame.shape[0]*2-200
        frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        # Inside the loop over detected faces
        for (x, y, w, h) in faces:
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = frame[y:y + h, x:x + w]

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = frame[y:y + h, x:x + w]
            predicted_right_eye_label_index = None  # Initialize variable outside loop
            predicted_left_eye_label_index = None
            # Detect left eye within the face region
            left_eyes = left_eye_cascade.detectMultiScale(roi_gray)
            for (ex, ey, ew, eh) in left_eyes:
                roi_left_eye = roi_gray[ey:ey + eh, ex:ex + ew]
                roi_left_eye = cv2.resize(roi_left_eye, (48, 48))  # Resize to (48, 48)
                roi_left_eye = cv2.cvtColor(roi_left_eye, cv2.COLOR_GRAY2RGB)  # Convert to RGB
                roi_left_eye = np.array(roi_left_eye) / 255.0
                result_left_eye = model.predict(roi_left_eye[np.newaxis, ..., np.newaxis])
                predicted_left_eye_label_index = np.argmax(result_left_eye)

                # Perform action based on predicted left eye state
                # Perform action based on predicted right eye state
                if predicted_left_eye_label_index == 2:  # Closed eye
                    cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 0, 255),
                                  2)  # Red rectangle for closed eye
                elif predicted_left_eye_label_index == 3:  # Open eye
                    cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0),
                                  2)  # Green rectangle for open eye

            # Detect right eye within the face region
            right_eyes = right_eye_cascade.detectMultiScale(roi_gray)
            for (ex, ey, ew, eh) in right_eyes:
                roi_right_eye = roi_gray[ey:ey + eh, ex:ex + ew]
                roi_right_eye = cv2.resize(roi_right_eye, (48, 48))  # Resize to (48, 48)
                roi_right_eye = cv2.cvtColor(roi_right_eye, cv2.COLOR_GRAY2RGB)  # Convert to RGB
                roi_right_eye = np.array(roi_right_eye) / 255.0
                result_right_eye = model.predict(roi_right_eye[np.newaxis, ..., np.newaxis])
                predicted_right_eye_label_index = np.argmax(result_right_eye)

                if predicted_right_eye_label_index == 2:  # Closed eye
                    cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 0, 255),
                                  2)  # Red rectangle for closed eye
                elif predicted_right_eye_label_index == 3:  # Open eye
                    cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0),
                                  2)  # Green rectangle for open eye

                # Display drowsiness indication above the face box
                if (predicted_right_eye_label_index == 2 and predicted_left_eye_label_index == 2):
                    cv2.putText(frame, 'Drowsiness Detected', (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                else:
                    cv2.putText(frame, 'NO Drowsiness Detected', (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Detect mouth within the face region
            #mouths = face_cascade.detectMultiScale(roi_gray)

        for (x, y, w, h) in faces:
                # Extract the region of interest (mouth) from the grayscale frame
                roi_mouth_gray = frame_gray[y:y + h, x:x + w]

                # Resize the mouth region to (48, 48)
                roi_mouth_gray_resized = cv2.resize(roi_mouth_gray, (48, 48))

                # Expand dimensions to make it compatible with model input shape
                roi_mouth_gray_resized = np.expand_dims(roi_mouth_gray_resized, axis=-1)

                # Normalize the pixel values to the range [0, 1]
                roi_mouth_gray_resized = roi_mouth_gray_resized / 255.0

                # Make prediction using the model
                result_mouth = model4.predict(np.expand_dims(roi_mouth_gray_resized, axis=0))
                result_age = model6.predict(np.expand_dims(roi_mouth_gray_resized, axis=0))
                # Get predicted mouth label index
                predicted_mouth_label_index = np.argmax(result_mouth)
                predicted_age_label_index = np.argmax(result_age)
                if predicted_age_label_index==0:
                    cv2.putText(frame, 'Age: Young(0-15)', (x+130, y - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 139, 139), 2)
                elif predicted_age_label_index==1:
                    cv2.putText(frame, 'Age: Middle(16-45)', (x+130, y - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (139, 0, 139), 2)
                else:
                    cv2.putText(frame, 'Age: Old(46-70) ', (x+130, y - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (139, 139, 0), 2)

                # Perform action based on predicted mouth state (yawning)
                if predicted_mouth_label_index == 0:
                    cv2.putText(frame, 'Gender: Male', (x, y - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (139, 0, 0), 2)
                else:
                    cv2.putText(frame, 'Gender: Female', (x, y - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 139), 2)

                if t==0:
                    alarm_thread = threading.Thread(target=play_greetings(predicted_mouth_label_index))
                    # Start the thread to play the alarm sound
                    alarm_thread.start()
                    t+=1

            # Perform emotion analysis
                face_roi = gray[y:y + h, x:x + w]
                if face_roi.shape[0] > 0 and face_roi.shape[1] > 0:
                    face_roi = cv2.resize(face_roi, (48, 48))
                    face_roi = face_roi.astype('float32') / 255.0
                    face_roi = face_roi.reshape(1, 48, 48, 1)

                    prediction1 = model2.predict(face_roi)[0]
                    prediction2 = model3.predict(face_roi)[0]

                    predicted1 = np.argmax(prediction1)
                    predicted2 = np.argmax(prediction2)

                # Determine frame color based on sentiment prediction
                if mapper1[predicted1] == "happy":
                    text = "Happy"
                    frame_color = (0, 255, 0)  # Bright Green
                elif mapper1[predicted1] == "surprised":
                    text = "Surprised"
                    frame_color = (128, 0, 255)  # Bright Pink
                elif (mapper1[predicted1] == "sad" or (
                        mapper1[predicted1] == "neutral" and mapper2[predicted2] == "sad")):
                    text = "Sad"
                    frame_color = (200, 126, 126)  # Bright Pink
                elif mapper1[predicted1] == "angry":
                    text = "Angry"
                    frame_color = (0, 0, 255)  # Bright Red
                elif mapper1[predicted1] == "neutral" and mapper2[predicted2] == "neutral":
                    text = 'Neutral'
                    frame_color = (255, 0, 0)  # Bright purple

                # Display the face with sentiment prediction and colored frame
                cv2.rectangle(frame, (x, y), (x + w, y + h), frame_color, 2)
                cv2.putText(frame, f'Emotion: {text}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            frame_color, 2)

                # Extract the region of interest (mouth) from the grayscale frame
                roi_gray = gray[y:y + h, x:x + w]

                # Resize the mouth region to the desired input dimensions of the yawn detection model
                yawn_input_shape = (48, 48)  # Specify the input dimensions expected by the yawn detection model

                # Resize the mouth region to the desired input dimensions of the yawn detection model
                roi_resizedd = cv2.resize(roi_gray, yawn_input_shape)

                # Convert the resized mouth region to RGB (if necessary)
                roi_rgbb = cv2.cvtColor(roi_resizedd,
                                        cv2.COLOR_GRAY2RGB)  # Convert from grayscale to RGB if necessary

                # Normalize the pixel values to the range [0, 1]
                roi_normalizedd = roi_rgbb / 255.0

                # Make prediction using the model for yawn detection
                result1 = model_mask.predict(np.expand_dims(roi_normalizedd, axis=0))
                result1 = np.argmax(result1)

                if result1 == 0:
                    cv2.putText(frame, 'Mask weared incorrect', (x, y - 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 139),
                                2)
                elif result1 == 1:
                    cv2.putText(frame, 'Mask weared', (x, y - 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (139, 0, 139), 2)
                else:
                    cv2.putText(frame, 'No Mask weared', (x, y - 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (139, 0, 0), 2)

                roi_resized = cv2.resize(roi_gray, yawn_input_shape)

                # Convert the resized mouth region to RGB (if necessary)
                roi_rgb = cv2.cvtColor(roi_resized,
                                           cv2.COLOR_GRAY2RGB)  # Convert from grayscale to RGB if necessary

                # Normalize the pixel values to the range [0, 1]
                roi_normalized = roi_rgb / 255.0

                # Make prediction using the model for yawn detection
                result = model_yawn.predict(np.expand_dims(roi_normalized, axis=0))

                if result < 0.5 and (result1==2):
                    cv2.putText(frame, 'Yawning detected', (x, y - 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 139),
                                    2)
                elif result > 0.5 and (result1==2):
                    cv2.putText(frame, 'No Yawning detected', (x, y - 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (139, 0, 0), 2)

                else:
                    cv2.putText(frame, "Can't detect yawning", (x, y - 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (139, 0, 0),
                                2)
                """
                if predicted_right_eye_label_index == 2 and predicted_left_eye_label_index == 2:
                    print("Changing to autonoums mode")
                    # Play the alarm sound (replace 'alarm.mp3' with your mp3 file)
                    # Create a thread to play the alarm sound
                    alarm_thread = threading.Thread(target=play_alarm_sound)
                    # Start the thread to play the alarm sound
                    alarm_thread.start()
                    #objectdetection()
                if mapper1[predicted1] == "angry" or mapper1[predicted1] == "sad" or (
                        mapper1[predicted1] == "neutral" and mapper2[predicted2] == "sad"):
                    print("Please change to the autounoums mode!")
                    #alarm_thread = threading.Thread(target=play_alarm)
                    # Start the thread to play the alarm sound
                    #alarm_thread.start()"""

        cv2.imshow('Video Analysis', frame)

        if cv2.waitKey(1) & 0xFF == ord('q') :
            break

    capture.release()
    cv2.destroyAllWindows()

# Create the GUI for the user
root = tk.Tk()
root.title("Welcome To Alex, Your AI Assistant")
root.geometry("500x700")
#root.resizable(False, False)

task_list = []

# Add button to perform video analysis
video_button = tk.Button(root, text="AI System", font="arial 20 bold", width=20, bg="#32405b", fg="#fff", bd=0, command=perform_video_analysis)
video_button.pack(pady=30)

# Function to resize image
def resize_image(image_path, output_path, width, height):
    img = cv2.imread(image_path)
    resized_img = cv2.resize(img, (width, height))
    cv2.imwrite(output_path, resized_img)

#Add task function
def addTask():
    task=task_entry.get()
    task_entry.delete(0,"end")

    if task:
        with open("tasklist.txt","a") as taskfile:
            # Perform sentiment analysis
            sentiment = single_response(task.lower())
            # Display feedback and sentiment analysis
            feedback = f"{task} - Sentiment: {sentiment}"
            taskfile.write(feedback + "\n")
            # Save the task to the database
            save_to_database(feedback)
        task_list.append(feedback)
        listbox.insert("end", feedback)

# Delete task function
def deleteTask():
    global task_list
    task=str(listbox.get("anchor"))
    if task in task_list:
        task_list.remove(task)
        with open("tasklist.txt","w") as taskfile:
            for task in task_list:
                taskfile.write(task+"\n")
        listbox.delete("anchor")

# Set the background color for the main window
root.configure(bg="#CFC3C8")  # Use your desired background color

# Add a title heading
heading=tk.Label(root, text="Please give us your Feedback🥰 ", font="Arial 15 bold ", bg="#CFC3C8")
heading.place(x=5, y=140)

#Main Entry frame
frame=tk.Frame(root, width=500, height=50, bg="white")
frame.place(x=0, y=180)

#Insert string
task= tk.StringVar()
task_entry=tk.Entry(frame, width=18, font="Arial 20 bold", bd=0)
task_entry.place(x=10, y=7)
task_entry.focus()

# The add button to add your tasks for the list
button=tk.Button(frame, text="ADD", font="arial 20 bold", width=6, bg="#32405b", fg="#fff", bd=0, command=addTask)
button.place(x=400,y=0)

# Create the listbox
frame1= tk.Frame(root,bd=3, width=500, height=280, bg="#32405b")
frame1.pack(pady=(150,0), fill="both", expand=True)

listbox= tk.Listbox(frame1, font=("arial", 12), width=40, height=16, bg="#32405b", fg="white", cursor="hand2", selectbackground="#5a95ff")
listbox.pack( side="left", fill="both", padx=2, expand=True)

#Scroll bar for al big list in vertical
scroll_bar=tk.Scrollbar(frame1, orient="vertical")
scroll_bar.pack(side="right", fill="y", expand=False)
listbox.config(yscrollcommand=scroll_bar.set)
scroll_bar.config(command=listbox.yview)

#Scroll bar for al big list in horizontal
scroll_bar2=tk.Scrollbar(listbox, orient="horizontal")
scroll_bar2.pack(side="bottom", fill="x", expand=False)
listbox.config(xscrollcommand=scroll_bar2.set)
scroll_bar2.config(command=listbox.xview)

#To delete an added task
Delete_Iamge=tk.PhotoImage(file="delete.png")
Delete_Label=tk.Button(root,image=Delete_Iamge, bd=0, bg="#CFC3C8", command=deleteTask)
Delete_Label.pack(side="bottom", pady=13)

root.mainloop()

"""
if __name__ == "__main__":
    print('Specify purpose')
    print("1.Image analysis")
    print("2.Video analysis")
    print("3.FeedBacks")
    inp = int(input("choose a number: "))
    if inp == 1:
        img_path = 'img_3.png'
        image_analysis(img_path)
    elif inp == 2:
        perform_video_analysis()
    elif inp == 3 :
        review=input("Give us your feedback: ")
        single_response(review.lower())
    else:
        print("Unrecognized Order")
        """