import cv2
import numpy as np
import os
import base64  
import paho.mqtt.client as mqtt
import time
import json

mqtt_broker = "mqttvcloud.innoway.vn"
mqtt_topic = "recognition"
control_topic = "camera/control"
mqtt_topic_camera="pub/camera"
username = "cde"
password = "jp4jfwJHNXaql5gJ9xZzF8PNJm7oZ2ND"

recognizer = cv2.face.LBPHFaceRecognizer_create()

recognizer.read('trainer/trainer.yml')

cascadePath = r"D:\Project\Face_detect\view\haarcascade_frontalface_default.xml"

faceCascade = cv2.CascadeClassifier(cascadePath)

font = cv2.FONT_HERSHEY_SIMPLEX

names = ['Paula', 'Ngoc', 'Ilza', 'Z', 'W']

camera_active = False

def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("Kết nối MQTT thành công!")
        mqtt_client.subscribe(control_topic)
    else:
        print("Kết nối MQTT thất bại với mã lỗi:", rc)

def on_message(client, userdata, message):
    global camera_active
    payload = message.payload.decode('utf-8')
    print(f"Nhận tin nhắn: {payload} từ topic: {message.topic}")  
    if message.topic == control_topic:
        if payload == '1':
            print("Camera started")
            camera_active = True
        elif payload == '0':
            print("Camera stopped")
            camera_active = False

mqtt_client = mqtt.Client()
mqtt_client.username_pw_set(username, password)

mqtt_client.on_connect = on_connect
mqtt_client.on_message = on_message

mqtt_client.connect(mqtt_broker)

mqtt_client.loop_start()

cam = None
last_pub_time = time.time()
publish_interval = 5

while True:
    if camera_active:
        if cam is None:  
            cam = cv2.VideoCapture(0)
            cam.set(3, 600) 
            cam.set(4, 500) 
            minW = 0.1 * cam.get(3)
            minH = 0.1 * cam.get(4)

        ret, img = cam.read()
        
        if not ret:
            print("Không thể lấy khung hình từ camera.")
            break

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(int(minW), int(minH)),
        )

        recognized_id = "unknown"
        
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2) 

            id, confidence = recognizer.predict(gray[y:y + h, x:x + w])

            if confidence < 100:
                recognized_id = names[id] 
                confidence = "{0}%".format(round(100 - confidence))
            else:
                recognized_id = "unknown"
                confidence = "0%"

            cv2.putText(img, str(recognized_id), (x + 5, y - 5), font, 1, (255, 255, 255), 2)
            cv2.putText(img, str(confidence), (x + 5, y + h - 5), font, 1, (255, 255, 0), 1)

        current_time = time.time()
        if current_time - last_pub_time >= publish_interval:
            if recognized_id == "unknown":
                mqtt_client.publish(mqtt_topic, 2) 
            else:
                mqtt_client.publish(mqtt_topic, 3) 
            last_pub_time = current_time

        _, buffer = cv2.imencode('.jpg', img)
        jpg_as_text = base64.b64encode(buffer).decode('utf-8')

        data = {
            'image': jpg_as_text,
            'name': recognized_id
        }

        json_data = json.dumps(data)

        mqtt_client.publish(mqtt_topic_camera, json_data)

        cv2.imshow('nhan dien khuon mat', img)

    else:
        if cam is not None: 
            cam.release()  
            cam = None 

    k = cv2.waitKey(10) & 0xff
    if k == 27:
        break

print("\n [INFO] Thoát")

if cam is not None:
    cam.release()
cv2.destroyAllWindows()

mqtt_client.loop_stop()
mqtt_client.disconnect()
