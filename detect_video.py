import argparse
import cv2
import numpy as np


def load_model(weights_path: str, config_path: str, classes_path: str):
    """Load pre-trained network weights, configuration, and custom class labels."""
    with open(classes_path, "r", encoding="utf-8") as f:
        classes = [line.strip() for line in f.readlines()]

    net = cv2.dnn.readNet(weights_path, config_path)
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    return net, output_layers, classes


def detect_and_localize(frame, net, output_layers, classes, conf_threshold=0.5, nms_threshold=0.4):
    """Run forward pass on a single video frame and draw localized bounding boxes."""
    height, width, _ = frame.shape
    blob = cv2.dnn.blobFromImage(frame, scalefactor=1 / 255.0, size=(416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)

    boxes, confidences, class_ids = [], [], []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = int(np.argmax(scores))
            confidence = float(scores[class_id])

            if confidence > conf_threshold:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(confidence)
                class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i]
            label = f"{classes[class_ids[i]]}: {confidences[i]:.2f}"
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, max(y - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return frame


def process_video(video_source, weights_path, config_path, classes_path, conf_threshold):
    """Read video stream frame-by-frame and display localized detections."""
    net, output_layers, classes = load_model(weights_path, config_path, classes_path)
    cap = cv2.VideoCapture(video_source)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        annotated_frame = detect_and_localize(frame, net, output_layers, classes, conf_threshold)
        cv2.imshow("Custom Object Detection & Localization", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Custom Object Detection in Videos")
    parser.add_argument("--video", type=str, default="sample_video.mp4", help="Path to input video file")
    parser.add_argument("--weights", type=str, default="custom_model.weights", help="Path to model weights")
    parser.add_argument("--config", type=str, default="custom_model.cfg", help="Path to model config")
    parser.add_argument("--classes", type=str, default="classes.names", help="Path to class names file")
    parser.add_argument("--confidence", type=float, default=0.5, help="Minimum detection confidence")
    args = parser.parse_args()

    process_video(args.video, args.weights, args.config, args.classes, args.confidence)
