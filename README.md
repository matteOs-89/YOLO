# YOLO
y
What is YOLO (You Only Look Once)?

YOLO is real time object detection algorithm, that uses fully convolutional Neural Networks and bounding box to perform classification and predict the class probability of the object present in each given image or frame grid. Therefore, it can be said that yolo models are used for object classifications and localization.



YOLO networks are faster and computational more cost effective than some of famous older object detection models such as Region with convolutional neural networks (R CNN), Fast R CNN and Faster CNN, reason been, YOLO models use only look once technique. This means the model only looks at the image once to localize the object present within and compute it’s confidence score. The confidence score is how sure the model is that an object is present and how certain it is of the class of the object in the grid box. YOLO predicts multiple bounding boxes per grid cell.


YOLO models are non-maximum suppression (NMS). 
NMS identifies and remove redundant, incorrect bounding boxes, overlapping bounding boxes and outputs a single bounding box for each object in the image. This in return improves the model performance and accuracy.

