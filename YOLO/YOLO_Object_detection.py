import cv2
import matplotlib.pyplot as plt
import numpy as np

"""
In this tutorial we use yolov4 algorithm to do object classification and localization. 

"""

image = cv2.imread("/Users/eseosa/Desktop/man-driving-a-small-car-BA0GEA.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


height, width = image.shape[:2]

blob_image = cv2.dnn.blobFromImage(image, 1/255.0,(416,416), swapRB=True, crop=False)
print(blob_image.shape)


with open("/Users/eseosa/Downloads/YOLO-3-OpenCV/yolo-coco-data/coco.names") as f:
    coco_labels = [line.strip() for line in f] #Getting class label from coco datasets


#print(coco_labels) #check dataset

# Retrieved pretrained YOLOV3 Weights and Parameters

yolo_V4_model = cv2.dnn.readNetFromDarknet("/Users/eseosa/Desktop/YOLO-4-OpenCV/yolov4.cfg",
                                             "/Users/eseosa/Desktop/YOLO-4-OpenCV/yolov4.weights")


model_layers_names = yolo_V4_model.getLayerNames()
#print(model_layers_names)

# Getting needed layers for yolov4
output_layer = [model_layers_names[i -1] for i in yolo_V3_model.getUnconnectedOutLayers()] #Get yolo layers

threshold = 0.3
probability_threshold = 0.5

color = np.random.randint(0, 255, size=(len(coco_labels), 3), dtype="uint8") #Random colors for each class

"""Forward Pass"""

yolo_V4_model.setInput(blob_image)

output_from_Model = yolo_V4_model.forward(output_layer)

class_idx = []
bounding_boxes = []
confidence_score = []

for result in output_from_Model:

    for objects_detected in result:

        score = objects_detected[5:]
        _class = np.argmax(score)

        current_confidence = score[_class]

        if current_confidence > probability_threshold:

            box = objects_detected[:4] * np.array([width, height, width, height])

            x_middle, y_middle, box_width, box_height = box
            x_min = int(x_middle - (box_width/2))
            y_min = int(y_middle - (box_height/2))

            bounding_boxes.append([x_min, y_min, int(box_width), int(box_height)])
            confidence_score.append(float(current_confidence))
            class_idx.append(_class)


NMS_results = cv2.dnn.NMSBoxes(bounding_boxes, confidence_score, probability_threshold, threshold)

count = 1

if len(NMS_results) > 0:

    for idx in NMS_results:
        print(f"Object {count}, {coco_labels[int(class_idx[idx])]}")
        count +=1

        x_min, y_min, box_width, box_height = bounding_boxes[idx][0], bounding_boxes[idx][1], bounding_boxes[idx][2], bounding_boxes[idx][3]

        box_color = color[class_idx[idx]].tolist()

        cv2.rectangle(image, (x_min,y_min), (x_min+box_width, y_min+box_height), box_color,2)

        text = f" {coco_labels[int(class_idx[idx])]}, {confidence_score[idx]}"

        cv2.putText(image, text, (x_min, y_min-8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)


    plt.imshow(image)
    plt.show()




