import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.title("🎯 Object Detection App")

# Load YOLO
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

with open("coco.names") as f:
    classes = f.read().strip().split("\n")

layer_names = net.getUnconnectedOutLayersNames()

def detect_image(image):
    frame = np.array(image)
    h, w, _ = frame.shape

    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True)
    net.setInput(blob)
    outputs = net.forward(layer_names)

    boxes, confidences, class_ids = [], [], []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5:
                cx, cy = int(detection[0]*w), int(detection[1]*h)
                bw, bh = int(detection[2]*w), int(detection[3]*h)
                x, y = int(cx-bw/2), int(cy-bh/2)

                boxes.append([x,y,bw,bh])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    if len(idxs) > 0:
        for i in idxs.flatten():
            x,y,w,h = boxes[i]
            label = classes[class_ids[i]]
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
            cv2.putText(frame,label,(x,y-10),
                        cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)

    return frame

uploaded_file = st.file_uploader("Upload an image", type=["jpg","png","jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image")

    if st.button("Detect Objects"):
        result = detect_image(image)
        st.image(result, caption="Detected Image")