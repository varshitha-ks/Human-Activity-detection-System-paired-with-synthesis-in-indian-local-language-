import tkinter as tk
from tkinter import filedialog, Label, Button, Radiobutton, StringVar
from collections import deque
import numpy as np
import cv2
from PIL import ImageFont, ImageDraw, Image
from googletrans import Translator

# Parameters class to include important paths and constants
class Parameters:
    def __init__(self, video_path):
        self.CLASSES = open("login-page-using-Python-and-MySQL-main/model/action_recognition_kinetics.txt").read().strip().split("\n")
        self.ACTION_RESNET = 'login-page-using-Python-and-MySQL-main/model/resnet-34_kinetics.onnx'
        self.VIDEO_PATH = video_path  # Video path is passed from the Tkinter file dialog
        self.SAMPLE_DURATION = 16  # Maximum deque size
        self.SAMPLE_SIZE = 112

# Function to run the human activity recognition
def human_activity_recognition(video_path, selected_language):
    param = Parameters(video_path)
    captures = deque(maxlen=param.SAMPLE_DURATION)

    # Load the human activity recognition model
    print("[INFO] loading human activity recognition model...")
    net = cv2.dnn.readNet(model=param.ACTION_RESNET)

    print("[INFO] accessing video stream...")
    vs = cv2.VideoCapture(param.VIDEO_PATH if param.VIDEO_PATH else 0)

    # Load font based on selected language
    font_size = 32
    if selected_language == "Kannada":
        font_path = "login-page-using-Python-and-MySQL-main/Tunga.ttf"  # Use Kannada font for Kannada
    else:
        font_path = "login-page-using-Python-and-MySQL-main/arial.ttf"  # Use default English font

    try:
        font = ImageFont.truetype(font_path, font_size)
    except IOError:
        print(f"Error: Could not load font at {font_path}. Ensure the path is correct.")
        return

    # Initialize the translator
    translator = Translator()

    while True:
        (grabbed, capture) = vs.read()
        if not grabbed:
            print("[INFO] no capture read from stream - exiting")
            break

        # Resize frame and append to deque
        capture = cv2.resize(capture, dsize=(550, 400))
        captures.append(capture)

        if len(captures) < param.SAMPLE_DURATION:
            continue

        # Create blob from images
        imageBlob = cv2.dnn.blobFromImages(captures, 1.0,
                                           (param.SAMPLE_SIZE, param.SAMPLE_SIZE),
                                           (114.7748, 107.7354, 99.4750),
                                           swapRB=True, crop=True)
        imageBlob = np.transpose(imageBlob, (1, 0, 2, 3))
        imageBlob = np.expand_dims(imageBlob, axis=0)

        # Forward pass through the model
        net.setInput(imageBlob)
        outputs = net.forward()

        # Get the label with maximum probability
        label = param.CLASSES[np.argmax(outputs)]

        # Translate label based on selected language
        if selected_language == "Kannada":
            try:
                translated_text = translator.translate(label, src='en', dest='kn').text
            except Exception as e:
                print(f"Error in translation: {e}")
                translated_text = label
        else:
            translated_text = label

        # Overlay translated text on video
        img_pil = Image.fromarray(capture)
        draw = ImageDraw.Draw(img_pil)
        draw.rectangle((0, 0, 300, 40), fill=(255, 255, 255))
        try:
            draw.text((10, 5), translated_text, font=font, fill=(0, 0, 0))
        except UnicodeEncodeError as e:
            print(f"Unicode error: {e}")
            draw.text((10, 5), label, font=font, fill=(0, 0, 0))

        # Convert back to OpenCV format and display
        capture = np.array(img_pil)
        cv2.imshow("Human Activity Recognition", capture)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    vs.release()
    cv2.destroyAllWindows()

# Function to browse and select a video file
def browse_video(selected_language):
    video_path = filedialog.askopenfilename(
        title="Select a Video File",
        filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv")]
    )
    if video_path:
        print(f"Selected video: {video_path}")
        human_activity_recognition(video_path, selected_language)
    else:
        print("No video file selected.")

# Create the Tkinter page
def create_gui():
    root = tk.Tk()
    root.title("Human Activity Recognition")

    # Set window size
    root.geometry("400x300")
    label = Label(root, text="Human Activity Recognition", font=("Arial", 20))
    label.pack(pady=10)
    # Language selection label
    label_lang = Label(root, text="Select Language:", font=("Arial", 12))
    label_lang.pack(pady=30)

    # StringVar to store the selected language
    selected_language = StringVar()
    # selected_language.set("Kannada")  # Default to English

    # Radiobuttons for language selection
    radio_en = Radiobutton(root, text="English", variable=selected_language, value="English", font=("Arial", 12))
    radio_kn = Radiobutton(root, text="Kannada", variable=selected_language, value="Kannada", font=("Arial", 12))

    radio_en.pack(pady=5)
    radio_kn.pack(pady=5)

    # Select Video Button
    select_button = Button(root, text="Select Video", command=lambda: browse_video(selected_language.get()), font=("Arial", 14), width=20)
    select_button.pack(pady=20)

    root.mainloop()

# Start the GUI
if __name__ == "__main__":
    create_gui()
