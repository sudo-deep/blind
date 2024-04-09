from ultralytics import YOLO
import cv2

model = YOLO("yolov8s.pt")

# cap = cv2.VideoCapture(1)
URL = "http://192.168.0.30"
cap = cv2.VideoCapture(URL + ":81/stream")


def vibrate(frequency):
    pass

while cap.isOpened():

    # cv2image = cv2.cvtColor(cap.read()[1], cv2.COLOR_BGR2RGBA)
    ret, frame = cap.read()
    if ret:
        
        result = model.predict(frame, conf=0.75)[0]
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy.numpy().tolist()[0])

            width = abs(x2 - x1)
            height = abs(y2 - y1)
            area = width * height
            print(area)

            # threshold: 200000 according to deepansh's laptop webcam
            stage = "high" if area > 200000 else "off"
            vibrate("high")
            
            name = str(int(box.cls))
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
            cv2.putText(frame, name, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
        cv2.imshow("test", frame)

    # Press Q on keyboard to exit 
    if cv2.waitKey(3) & 0xFF == ord('q'): 
        break


cap.release()
cv2.destroyAllWindows()