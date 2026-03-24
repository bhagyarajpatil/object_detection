import cv2
import numpy as np
import os
import tkinter as tk
from tkinter import filedialog, messagebox

# ==========================================
# FILE PATHS
# ==========================================
weights_path = r"C:\Users\bhagr\OneDrive\Documents\object_detecton\yolov3.weights"
config_path = r"C:\Users\bhagr\OneDrive\Documents\object_detecton\yolov3.cfg"
names_path = r"C:\Users\bhagr\OneDrive\Documents\object_detecton\coco.names"

# ==========================================
# LOAD YOLO
# ==========================================
print("🔄 Loading YOLO model...")
net = cv2.dnn.readNet(weights_path, config_path)

with open(names_path, "r") as f:
    classes = f.read().strip().split("\n")

layer_names = net.getUnconnectedOutLayersNames()
print("✅ Model Loaded!")

# ==========================================
# DETECTION FUNCTION
# ==========================================
def detect_objects(frame):
    height, width, _ = frame.shape

    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(layer_names)

    boxes, confidences, class_ids = [], [], []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    count_dict = {}
    total_count = 0

    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = classes[class_ids[i]]

            count_dict[label] = count_dict.get(label, 0) + 1
            total_count += 1

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # TOTAL
    cv2.putText(frame, f"Total: {total_count}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 0, 255), 2)

    # CLASS-WISE
    y_offset = 60
    for label, count in count_dict.items():
        cv2.putText(frame, f"{label}: {count}",
                    (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 0, 0), 2)
        y_offset += 25

    return frame

# ==========================================
# GUI FUNCTIONS
# ==========================================

def detect_image():
    path = filedialog.askopenfilename()
    if not path:
        return

    frame = cv2.imread(path)
    output = detect_objects(frame)

    cv2.imshow("Image Detection", output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def detect_video():
    path = filedialog.askopenfilename()
    if not path:
        return

    cap = cv2.VideoCapture(path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = detect_objects(frame)
        cv2.imshow("Video Detection", frame)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def detect_webcam():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = detect_objects(frame)
        cv2.imshow("Webcam Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# ==========================================
# GUI WINDOW
# ==========================================
root = tk.Tk()
root.title("Object Detection System")
root.geometry("400x300")
root.configure(bg="#1e1e1e")

tk.Label(root,
         text="Object Detection YOLO",
         font=("Arial", 16, "bold"),
         bg="#1e1e1e",
         fg="#00ffcc").pack(pady=20)

tk.Button(root, text="📷 Detect Image", width=20,
          bg="#007acc", fg="white",
          activebackground="#005f99",
          command=detect_image).pack(pady=5)

tk.Button(root, text="🎥 Detect Video", width=20,
          bg="#28a745", fg="white",
          activebackground="#1e7e34",
          command=detect_video).pack(pady=5)

tk.Button(root, text="📹 Webcam", width=20,
          bg="#ffc107", fg="black",
          activebackground="#e0a800",
          command=detect_webcam).pack(pady=5)

tk.Button(root, text="❌ Exit", width=20,
          bg="#dc3545", fg="white",
          activebackground="#a71d2a",
          command=root.quit).pack(pady=20)

root.mainloop()