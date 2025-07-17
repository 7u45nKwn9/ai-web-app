import cv2
import numpy as np
import yaml

# --- Class from yaml ---
with open("model/data.yaml", "r") as f:
    data = yaml.safe_load(f)
CLASSES = data["names"]  

# --- Preprocessing ---
def preprocess(image, img_size=640):
    resized = cv2.resize(image, (img_size, img_size))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    rgb = rgb.astype(np.float32) / 255.0
    tensor = np.transpose(rgb, (2, 0, 1))  # HWC → CHW
    tensor = np.expand_dims(tensor, axis=0)
    return tensor.astype(np.float32)

# --- Postprocessing for yolov8 ---
def postprocess_yolov8_raw(pred, orig_shape, model_resolution=(640, 640), conf_thres=0.1, model_name=None):
    boxes = []

    pred = pred[0]  # shape: (1, 5, 8400)
    if pred.ndim == 3 and pred.shape[0] == 1:
        pred = pred[0]  # (5, 8400)
    elif pred.shape[0] != 5:
        print("[ERROR] Unexpected shape after squeeze:", pred.shape)
        return boxes

    # Debug shape
    print("[DEBUG] Using pred shape:", pred.shape)

    h_orig, w_orig = orig_shape
    w_model, h_model = model_resolution
    dx = w_orig / w_model
    dy = h_orig / h_model

    x_center = pred[0]
    y_center = pred[1]
    width = pred[2]
    height = pred[3]
    obj_conf = pred[4]

    print("[DEBUG] Max objectness:", np.max(obj_conf))

    for i in range(pred.shape[1]):
        score = float(obj_conf[i])
        if score < conf_thres:
            continue

        xc, yc, w, h = x_center[i], y_center[i], width[i], height[i]
        x0 = int((xc - w / 2) * dx)
        y0 = int((yc - h / 2) * dy)
        x1 = int((xc + w / 2) * dx)
        y1 = int((yc + h / 2) * dy)
        boxes.append([x0, y0, x1, y1, round(score, 4), 0])

    if not boxes:
        print("[DEBUG] No boxes passed the confidence threshold.")

    return boxes


# ---NMS---
def non_max_suppression(boxes, iou_threshold=0.5):
    if not boxes:
        return []

    boxes = sorted(boxes, key=lambda x: x[4], reverse=True)  # sort by confidence
    selected_boxes = []

    while boxes:
        best = boxes.pop(0)
        selected_boxes.append(best)
        boxes = [box for box in boxes if iou(best, box) < iou_threshold]

    return selected_boxes


def iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)
    box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

    return inter_area / float(box1_area + box2_area - inter_area)

# --- Draw bounding boxes ---
def conf_to_color(conf):
    r = int(255 * (1 - conf))
    g = int(255 * conf)
    return (r, g, 0)

def draw_boxes(image, boxes, classes):
    for box in boxes:
        x1, y1, x2, y2, conf, cls_id = box
        label = f"{classes[cls_id].capitalize()} {round(conf * 100)}%" if classes else f"Class {cls_id} {round(conf * 100)}%"
        color = conf_to_color(conf)

        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        overlay = image.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        image = cv2.addWeighted(overlay, 0.2, image, 0.8, 0)

    return image

# --- Process Images ---
def process_image(
    frame,
    session,
    preprocess,
    postprocess,
    draw_boxes,
    model_name="polyp.onnx",
    model_resolution=(640, 640),
    classes=None
):
    import time
    start_total = time.time()

    input_tensor = preprocess(frame)

    start_inf = time.time()
    outputs = session.run(None, {session.get_inputs()[0].name: input_tensor})
    print("[DEBUG] Number of outputs:", len(outputs))
    for i, out in enumerate(outputs):
        print(f"[DEBUG] Output {i}: shape={out.shape}, dtype={out.dtype}")
        print("[DEBUG] Sample values:", out[:1])
    inference_time = (time.time() - start_inf) * 1000  # ms

    boxes = postprocess_yolov8_raw(outputs, frame.shape[:2], model_resolution)
    boxes = non_max_suppression(boxes, iou_threshold=0.5)
    result_image = draw_boxes(frame.copy(), boxes, classes)

    result_image = draw_boxes(frame.copy(), boxes, classes)

    total_time = (time.time() - start_total) * 1000  # ms

    print(f"Inference Time: {inference_time:.2f} ms")
    print(f"Total Time:     {total_time:.2f} ms")
    print(f"Overhead Time:  {total_time - inference_time:.2f} ms")
    print(f"Inference FPS:  {1000/inference_time:.2f} fps")
    print(f"Total FPS:      {1000/total_time:.2f} fps")

    return result_image
