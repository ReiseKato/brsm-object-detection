import cv2
import time

# Paths
cfg_path = "yolov4-tiny.cfg"
weights_path = "yolov4-tiny.weights"
names_path = "obj.names"

# Load class names
with open(names_path, "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Load network
net = cv2.dnn.readNetFromDarknet(cfg_path, weights_path)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)  # or DNN_TARGET_CUDA if built with CUDA

# Prepare image
image = cv2.imread("test.jpg")
(H, W) = image.shape[:2]
blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
net.setInput(blob)

# Run inference
start = time.time()
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]
outputs = net.forward(output_layers)
end = time.time()
print(f"Inference time: {end - start:.3f}s")

# Draw detections
conf_threshold = 0.25
nms_threshold = 0.4
boxes, confidences, class_ids = [], [], []

for output in outputs:
    for detection in output:
        scores = detection[5:]
        class_id = int(np.argmax(scores))
        confidence = scores[class_id]
        if confidence > conf_threshold:
            box = detection[0:4] * [W, H, W, H]
            (centerX, centerY, width, height) = box.astype("int")
            x = int(centerX - (width / 2))
            y = int(centerY - (height / 2))
            boxes.append([x, y, int(width), int(height)])
            confidences.append(float(confidence))
            class_ids.append(class_id)

# Apply NMS and draw boxes
idxs = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
for i in idxs.flatten():
    (x, y, w, h) = boxes[i]
    label = f"{classes[class_ids[i]]}: {confidences[i]:.2f}"
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(image, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

cv2.imwrite("result.jpg", image)
cv2.imshow("Detections", image)
cv2.waitKey(0)
