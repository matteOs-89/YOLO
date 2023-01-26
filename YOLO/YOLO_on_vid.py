import cv2
import matplotlib.pyplot as plt
import numpy as np

vid = cv2.VideoCapture("/Users/eseosa/Downloads/YOLO-3-OpenCV/videos/traffic-cars.mp4")

writer = None
h, w, = None, None

with open("/Users/eseosa/Downloads/YOLO-3-OpenCV/yolo-coco-data/coco.names") as f:
    coco_labels = [line.strip() for line in f] #Getting class label from coco datasets


yolo_V3_model = cv2.dnn.readNetFromDarknet("/Users/eseosa/Desktop/YOLO-4-OpenCV/yolov4.cfg",
                                             "/Users/eseosa/Desktop/YOLO-4-OpenCV/yolov4.weights")


model_layers_names = yolo_V3_model.getLayerNames()

output_layer = [model_layers_names[i -1] for i in yolo_V3_model.getUnconnectedOutLayers()]

threshold = 0.3
probability_threshold = 0.5

color = np.random.randint(0, 255, size=(len(coco_labels), 3), dtype="uint8") #Random colors for each class
frame = 0

while True:

    ret, frame = vid.read()

    if not ret:
        break

    if w is None or h is None:
        h, w = frame.shape[:2]

    

    blob_frame = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False) #[1, 3, w, h]

    yolo_V3_model.setInput(blob_frame)

    output_from_Model = yolo_V3_model.forward(output_layer)

    class_idx = []
    bounding_boxes = []
    confidence_score = []

    for result in output_from_Model:

        for vid_object_detected in result:

            score = vid_object_detected[5:]
            _class = np.argmax(score)

            current_confidence = score[_class]

            if current_confidence > probability_threshold:

                box = vid_object_detected[:4] * np.array([w, h, w, h])


                x_middle, y_middle, box_width, box_height = box
                x_min = int(x_middle - (box_width / 2))
                y_min = int(y_middle - (box_height / 2))

                bounding_boxes.append([x_min, y_min, int(box_width), int(box_height)])
                confidence_score.append(float(current_confidence))
                class_idx.append(_class)

    NMS_results = cv2.dnn.NMSBoxes(bounding_boxes, confidence_score, probability_threshold, threshold)

    count = 1

    if len(NMS_results) > 0:

        for idx in NMS_results:
            print(f"Object {count}, {coco_labels[int(class_idx[idx])]}")
            count += 1

            x_min, y_min, box_width, box_height = bounding_boxes[idx][0], bounding_boxes[idx][1], bounding_boxes[idx][
                2], bounding_boxes[idx][3]

            box_color = color[class_idx[idx]].tolist()

            cv2.rectangle(frame, (x_min, y_min), (x_min + box_width, y_min + box_height), box_color, 2)

            text = f" {coco_labels[int(class_idx[idx])]}, {confidence_score[idx]}"

            cv2.putText(frame, text, (x_min, y_min - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)


    cv2.namedWindow("traffic_vid", cv2.WINDOW_NORMAL)
    cv2.imshow("traffic_vid", frame)

    if cv2.waitKey(0) & 0xFF == ord("q"):
        break

    if writer is None:

        format_ = cv2.VideoWriter_fourcc(*"mp4v")

        writer = cv2.VideoWriter("/Users/eseosa/Downloads/YOLO-3-OpenCV/videos/results-traffic cars", format_, 30, (frame.shape[1], frame.shape[0]), True)



    writer.write(frame)

vid.release()
writer.release()
cv2.destroyAllWindows()




