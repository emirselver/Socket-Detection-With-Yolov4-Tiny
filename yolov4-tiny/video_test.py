import cv2
import numpy as np

try:
    net = cv2.dnn.readNet('model/yolov4-tiny-obj_last.weights', 'model/yolov4-tiny-obj.cfg')

    classes = ['SOCKET', 'FULL', 'EMPTY']

    cap = cv2.VideoCapture('test/input/test_video.mp4')

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    out = cv2.VideoWriter('test/output/(result) test_video.mp4', cv2.VideoWriter_fourcc(*'XVID'), fps, (frame_width, frame_height))

    print("Video işleniyor...")

    while True:
        ret, frame = cap.read()

        if not ret:
            break

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
                    center_x = int(detection[0] * frame.shape[1])
                    center_y = int(detection[1] * frame.shape[0])
                    width = int(detection[2] * frame.shape[1])
                    height = int(detection[3] * frame.shape[0])

                    x = int(center_x - width / 2)
                    y = int(center_y - height / 2)

                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([x, y, width, height])

        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        for i in indices:
            box = boxes[i]
            x, y, width, height = box
            class_id = class_ids[i]
            confidence = confidences[i]
            label = f'{classes[class_id]}: {confidence:.2f}'
            color = (255, 0, 0)

            if classes[class_id] == 'SOCKET':
                color = color
            if classes[class_id] == 'FULL':
                color = (0, 255, 0)
            if classes[class_id] == 'EMPTY':
                color = (0, 0, 255)

            cv2.rectangle(frame, (x, y), (x + width, y + height), color, 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        out.write(frame)

        # cv2.imshow('Socket Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print("Video işleme tamalandı.")

    cap.release()
    out.release()
    cv2.destroyAllWindows()

except Exception as ex:
    print(ex)
