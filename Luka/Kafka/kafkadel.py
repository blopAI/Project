import json
import numpy as np
import cv2
import threading
import redis
import kafka
import signal
from datetime import datetime
import time
from ultralytics import YOLO
import os
os.system('play -nq -t alsa synth {} sine {}'.format(0.2, 22050))



def thread_do_work():

    red = redis.Redis()

    con_topic = 'frame_noticifation'
    prod_topic = 'frame_detection'
    producer = kafka.KafkaProducer(bootstrap_servers='10.211.55.12:9092')

    consumer = kafka.KafkaConsumer(bootstrap_servers='10.211.55.12:9092', auto_offset_reset='earliest', group_id='grp_detection', consumer_timeout_ms=2000)
    consumer.subscribe([con_topic])

    model = YOLO("/home/parallels/Desktop/train21/weights/best.pt")

    frame_number = 0
    frame_skip = 8

    while True:
        for message in consumer:
            y = json.loads(message.value.decode("utf-8"))
            if y["type"] == "new_frame":
                frame_time = datetime.fromtimestamp(message.timestamp / 1000)
                curr_time = datetime.now()
                diff = (curr_time - frame_time).total_seconds()
                if diff < 2:
                    frame_temp = np.frombuffer(red.get("frame:latest"), dtype=np.uint8)
                    if (np.shape(frame_temp)[0] == 1229760):
                        frame_number += 1
                        if frame_skip == 0 or frame_number % frame_skip == 0:  
                            frame = frame_temp.reshape((480, 854, 3))
                            results = model.predict(frame)
                            for result in results:
                                names = result.names
                                preds_list = []
                                for box in result.boxes.data:
                                    class_id = int(box[5])
                                    class_name = names[class_id]
                                    if class_name == "curb":
                                        print("\a")
                                        # Posiljanje podatkov o okvirju v Redis
                                        preds_list.append(",".join([class_name] + [str(v) for v in box[:4].tolist()]))
                                        pred = [int(float(v)) for v in box[:4].tolist()]
                                        cv2.line(frame, (pred[0], pred[1]), (pred[2], pred[1]), (0, 0, 255), 5)
                                        cv2.line(frame, (pred[0], pred[3]), (pred[2], pred[3]), (0, 0, 255), 5)
                                        cv2.line(frame, (pred[0], pred[1]), (pred[0], pred[3]), (0, 0, 255), 5)
                                        cv2.line(frame, (pred[2], pred[1]), (pred[2], pred[3]), (0, 0, 255), 5)
                                        cv2.putText(frame, class_name, (pred[0], pred[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
                                future = producer.send(prod_topic, str.encode("|".join(preds_list)), timestamp_ms=round(time.time()*1000))
                                try:
                                    rm = future.get(timeout=10)
                                except kafka.KafkaError:
                                    pass
                            cv2.imshow("frame", frame)
                            cv2.waitKey(1)

            if event.is_set():
                break
        if event.is_set():
            break

    cv2.destroyAllWindows()

def sigint_handler(signum, frame):
    event.set()
    thread.join()
    exit(0)

signal.signal(signal.SIGINT, sigint_handler)

event = threading.Event()
thread = threading.Thread(target=lambda: thread_do_work())

if __name__ == "__main__":
    thread.start()
    input("Detecting...")
    event.set()
    thread.join()
import cv2
import threading
import redis
import kafka
import signal
import json
import time
from kafka import TopicPartition
from datetime import datetime
import numpy as np

# Global detection list
# TODO: Using globals is not the best solution, modify this
preds_list = []
preds_time = datetime.fromtimestamp(0)
lock = threading.Lock()

def thread_detection():
    global preds_list, preds_time

    # Kafka
    topic = 'frame_detection'
    consumer = kafka.KafkaConsumer(bootstrap_servers='10.211.55.12:9092', auto_offset_reset='earliest', consumer_timeout_ms=2000)
    consumer.subscribe([topic])

    while True:
    
        # Preberemo sporocilo iz kafke
        for message in consumer:

            # Dekodiramo string
            preds_str = message.value.decode("utf-8")
            print(preds_str)
           
            with lock:
                preds_list = preds_str.split("|") if len(preds_str) > 0 else []
                preds_time = datetime.fromtimestamp(message.timestamp / 1000)

            if event.is_set():
                break   

        if event.is_set():
            cv2.destroyAllWindows()
            break    

def thread_frames():
    global preds_list, preds_time

    # Redis
    red = redis.Redis()

    # Video
    frame = 0

    # Kafka
    topic = 'frame_noticifation'
    consumer = kafka.KafkaConsumer(bootstrap_servers='10.211.55.12:9092', auto_offset_reset='latest', group_id='grp_visualization', consumer_timeout_ms=2000)

    # Seek to the end of the topic partition
    topic_partition = TopicPartition(topic, 0)
    consumer.assign([topic_partition])
    consumer.seek_to_end(topic_partition)

    while True:

        # Potegnemo zadnje sporocilo iz protokola
        msg_data = None
        msgs = consumer.poll(timeout_ms=1000, max_records=1000)

        for tp, messages in msgs.items():
            if messages:
                latest_message = messages[-1]
                msg_data = json.loads(latest_message.value.decode("utf-8"))

        # Procesiranje zadnjega sporocila
        if msg_data:
          
            #print(f"Received message: {msg_data}")
            
            if msg_data["type"] == "new_frame":
                frame_time = datetime.fromtimestamp(latest_message.timestamp / 1000)
                curr_time = datetime.now()
                diff = (curr_time - frame_time).total_seconds()

                # Odstranimo stare frame
                if diff < 2:
                    frame_temp = np.frombuffer(red.get("frame:latest"), dtype=np.uint8)

                    # Convert image
                    if (np.shape(frame_temp)[0] == 1229760):
                        frame = frame_temp.reshape((480, 854, 3))

                    # Detekcija
                    if (curr_time - preds_time).total_seconds() < 5:
                        with lock:
                            for pred_str in preds_list:
                                class_name, *pred_coords = pred_str.split(",")
                                pred = [int(float(v)) for v in pred_coords]
                                # Dodamo ime robnika
                                cv2.putText(frame, class_name, (pred[0], pred[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
                                # zgornji rob
                                cv2.line(frame, (pred[0], pred[1]), (pred[2], pred[1]), (0, 0, 255), 5)
                                # spodnji rob
                                cv2.line(frame, (pred[0], pred[3]), (pred[2], pred[3]), (0, 0, 255), 5)
                                # levo
                                cv2.line(frame, (pred[0], pred[1]), (pred[0], pred[3]), (0, 0, 255), 5)
                                # desno
                                cv2.line(frame, (pred[2], pred[1]), (pred[2], pred[3]), (0, 0, 255), 5)

                    cv2.imshow("frame", frame)
                    cv2.waitKey(1)
                   

            if event.is_set():
                break

        if event.is_set():
            cv2.destroyAllWindows()
            break


def sigint_handler(signum, frame):
    event.set()
    thread_frm.join()
    thread_det.join()
    exit(0)

signal.signal(signal.SIGINT, sigint_handler)

event = threading.Event()
thread_frm = threading.Thread(target=lambda: thread_frames())
thread_det = threading.Thread(target=lambda: thread_detection())


if __name__ == "__main__":
    thread_frm.start()
    thread_det.start()
    input("Press CTRL+C or Enter to stop visualization...")
    event.set()
    thread_frm.join()
    thread_det.join()

import numpy as np
import cv2
import threading
import redis
import signal
import time
import kafka
import json
import ffmpeg
import wave as wav

def thread_produce():
    # Redis
    red = redis.Redis()

    # Video
    input = "data/video8.mp4"
    vc = cv2.VideoCapture(input)
    fps = 30

    # Kafka
    topic = 'frame_noticifation'
    producer = kafka.KafkaProducer(bootstrap_servers='10.211.55.12:9092')
	
    while True:
        t_start = time.perf_counter()
        ret, frame = vc.read()

        # Skocimo nazaj na zacetek
        if not ret:
            vc.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        # dodamo rame v redis
        red.set("frame:latest", np.array(frame).tobytes())

        # Posljemo sporocilo o novem framu prek kafke
        message = {
            "type": "new_frame",
            "frame_number": int(vc.get(cv2.CAP_PROP_POS_FRAMES)),
            "timestamp_ms": round(time.time() * 1000),
        }
        future = producer.send(topic, json.dumps(message).encode("utf-8"))

        # Pocakamo, da je sporocilo odposlano cez Kafko
        try:
            rm = future.get(timeout=10)
        except kafka.KafkaError:
            pass

        t_stop = time.perf_counter()
        t_elapsed = t_stop - t_start
        t_frame = 1000 / fps / 1000
        t_sleep = t_frame - t_elapsed
        if t_sleep > 0:
            time.sleep(t_sleep)

        if event.is_set():
            vc.release()
            break    

def sigint_handler(signum, frame):
    event.set()
    thread.join()
    exit(0)

signal.signal(signal.SIGINT, sigint_handler)

event = threading.Event()
thread = threading.Thread(target=lambda: thread_produce())

if __name__ == "__main__":
    thread.start()
    input("Press CTRL+C or Enter to stop producing...")
    event.set()
    thread.join()
