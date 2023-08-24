import cv2
import numpy as np
import os

try:
    net = cv2.dnn.readNet('model/yolov4-tiny-obj_last.weights', 'model/yolov4-tiny-obj.cfg')
    classes = ['SOCKET', 'FULL', 'EMPTY']
    class_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    image_path = 'test/input/test_img.jpg'
    output_folder = 'test/output/'

    frame = cv2.imread(image_path)
    height, width = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(frame, scalefactor=1 / 255, size=(416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layer_names = net.getUnconnectedOutLayersNames()
    outputs = net.forward(layer_names)

    class_ids = []
    confidences = []
    boxes = []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.8:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    for i in indices:
        box = boxes[i]
        x, y, w, h = box
        class_id = class_ids[i]
        confidence = confidences[i]
        class_name = classes[class_id]
        color = class_colors[class_id]

        label = f"{class_name}: {confidence:.2f}"
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    marked_image_path = '(result) ' + os.path.basename(image_path)
    saved = os.path.join(output_folder, marked_image_path)
    cv2.imwrite(saved, frame)

    print("Tespit işlemi tamalandı.")
    print(f"Kaydedilen dosya adı: {marked_image_path}")

except Exception as ex:
    print(ex)