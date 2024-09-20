from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import shutil
import os
from moviepy.editor import VideoFileClip
from moviepy.video.fx.all import speedx
import cv2
import os
import matplotlib.pyplot as plt
from PIL import Image
from inference_sdk import InferenceHTTPClient
from easyocr import Reader
import re
import cv2
from huggingface_hub import login
from openai import OpenAI

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)


@app.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    file_location = f"{UPLOAD_DIR}/{file.filename}"
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    image_files = split_to_frames(file_location)
    final_results = analyze_the_video(image_files)
    if len(incident_image) == 0:
        answer = "No Incident Detected in the video"
    else:
        text = extract_text()
        time = extract_time_from_text(text=text)
        date = extract_date_from_text(text)
        prompt = """
            You will create an incident report based on the following sequence of events. Please analyze each event step by step and then compile the details into a structured report.

            Sequence of Events:
            """
        for event in final_results:
            chain_of_thought_prompt += f"\n- {event}"
            chain_of_thought_prompt += f"\n Date: {date}"
            chain_of_thought_prompt += f"\n Time: {time}"
            chain_of_thought_prompt += "\n\nLet's break this down step by step. First, indicate the date and time of the incident, then give a brief description of incident by explaining the sequence of event. Also describe the root cause, any injuries/goods damage, actions taken, and any recommendations."

        answer = ""
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,  # Number of tokens as needed
            n=1,  # Number of responses to generate
            stop=None,
            temperature=1.2,  # The temperature is adjusted for more creative or focused responses
            seed=0,
            stream=True,
        )
        for chunk in response:
            if chunk.choices[0].delta.content is not None:
                answer = answer + chunk.choices[0].delta.content
    return JSONResponse(content=answer)


incident_image = []


# Analyses the video using Roboflow Inference
def analyze_the_video(image_files):
    CLIENT = InferenceHTTPClient(
        api_url="https://detect.roboflow.com", api_key="9JoiXLYWniNRH8V9fZGf"
    )

    results = []
    final_results = []
    object = {}
    for i, image_path in enumerate(image_files):
        result = CLIENT.infer(image_path, model_id="my_project-1ecqf/1")
        results.append(result)
        for j in range(len(result["predictions"])):
            if "Incident" in result["predictions"][j]["class"]:
                incident_image.append(image_path)
            object["frame"] = i
            object["x"] = result["predictions"][j]["x"]
            object["y"] = result["predictions"][j]["y"]
            object["class"] = result["predictions"][j]["class"]
            final_results.append(object)
            object = {}
        print(result)
    unique_items = []
    for item in final_results:
        if item["class"] not in unique_items:
            unique_items.append(item["class"])

    return str(unique_items)


# Detect and return the date of the incdent
def extract_date_from_text(text):
    pattern = r"\d{2}-\d{2}-\d{4}"
    matches = re.findall(pattern, text)
    if matches:
        print(matches[0])
        return str(matches[0])
    return None


# Detect and return the time of incident
def extract_time_from_text(text):
    pattern = r"\d{2}\.\d{2}\.\d{2}"
    pattern1 = r"\d{2}\.\d{2}"
    matches = re.findall(pattern, text)
    matches1 = re.findall(pattern1, text)
    if matches:
        print("time: ", matches[0])
        return str(matches[0])
    return str(matches1[0])


# EasyOCR to extract text from the video
def extract_text():
    image_path = incident_image[0]
    image = cv2.imread(image_path)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    reader = Reader(["en"])  # Read English text
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blurred_image = cv2.medianBlur(gray_image, 3)
    read = reader.readtext(blurred_image)
    text = read[0][1]
    print("text: ", text)  # Get the text and confidence score
    return text


# Frame extraction
def split_to_frames(file_path: str):
    output_dir = "output_frames"
    os.makedirs(output_dir, exist_ok=True)

    # Load the video
    video = cv2.VideoCapture(file_path)
    image_files = []
    # Frame counter
    frame_number = 0
    sf = 0
    while True:
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

        # Read a frame from the video
        ret, frame = video.read()

        # If the frame was not retrieved, break the loop
        if not ret:
            break

        # Construct the filename for the frame
        frame_filename = os.path.join(output_dir, f"frame_{sf:05d}.jpg")
        image_files.append(frame_filename)
        # Save the frame as an image file
        cv2.imwrite(frame_filename, frame)

        # Increment the frame counter
        frame_number += 100
        sf += 1

    # Release the video capture object
    video.release()
    return image_files


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
