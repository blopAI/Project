from ultralytics import YOLO
import cv2
import numpy as np
import supervision as sv

FRAMERATE = 30
CONFIDENCE = 0.7
MOVEMENT_THRESHOLD = 3
IS_MOVING = False
SIGN_DETECTED = False


class Driver:
    def __init__(self, name: str, surname: str, age: int, is_disabled: bool) -> None:
        self.name = name
        self.surname = surname
        self.age = age
        self.is_disabled = is_disabled


def box_label(image, box, label='', color=(128, 128, 128), txt_color=(255, 255, 255)):
  lw = max(round(sum(image.shape) / 2 * 0.003), 2)
  p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
  cv2.rectangle(image, p1, p2, color, thickness=lw, lineType=cv2.LINE_AA)
  if label:
    tf = max(lw - 1, 1)  # font thickness
    w, h = cv2.getTextSize(label, 0, fontScale=lw / 3, thickness=tf)[0]  # text width, height
    outside = p1[1] - h >= 3
    p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
    cv2.rectangle(image, p1, p2, color, -1, cv2.LINE_AA)  # filled
    cv2.putText(image,
                label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2),
                0,
                lw / 3,
                txt_color,
                thickness=tf,
                lineType=cv2.LINE_AA)


def plot_bboxes(image, boxes, labels=[], colors=[], score=True, conf=None):
    if labels == []:
        labels = {0: u'__background__', 1: u'sign'}
    if colors == []:
        colors = [(89, 161, 197),(67, 161, 255),]
  
    for box in boxes:
        #add score in label if score=True
        if score :
            label = labels[int(box[-1])+1] + " " + str(round(100 * float(box[-2]),1)) + "%"
        else :
            label = labels[int(box[-1])+1]

        if conf :
            if box[-2] > conf:
                color = colors[int(box[-1])]
                box_label(image, box, label, color)
        else:
            color = colors[int(box[-1])]
            box_label(image, box, label, color)

    #show image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


driver1 = Driver('Gasper', 'Hribersek', 21, False)

model = YOLO('train7/weights/best.pt') # model za znake invalidi
 
cap = cv2.VideoCapture('data/videos/vid_record_2023_5_6_6.mp4')

whiteBackground = np.full((1280, 720, 3), 255, dtype = "uint8")

out_text = ''

counter = 0

# prev_p1 = (int(boxes[0,0]), int(boxes[0,1]))
# prev_p2 = (int(boxes[0,2]), int(boxes[0,3]))
prev_p1 = (0, 0)
prev_p2 = (0, 0)

while cap.isOpened():
    ret,frame = cap.read()
    if not ret:
        continue

    results = model(frame)
    # names = results.names
    # pred = results.xyxy[0].numpy()

    # Top
    # cv2.line(frame, (pred[0], pred[1]), (pred[2], pred[1]), (0, 0, 255), 5)
    # Bottom
    # cv2.line(frame, (pred[0], pred[3]), (pred[2], pred[3]), (0, 0, 255), 5)
    # Left
    # cv2.line(frame, (pred[0], pred[1]), (pred[0], pred[3]), (0, 0, 255), 5)
    # Right
    # cv2.line(frame, (pred[2], pred[1]), (pred[2], pred[3]), (0, 0, 255), 5)

    boxes = results[0].boxes.boxes

    if counter % 30 == 0:
        p1 = (int(boxes[0,0]), int(boxes[0,1]))
        p2 = (int(boxes[0,2]), int(boxes[0,3]))

        p1_diff = max(abs(prev_p1[0] - p1[0]), abs(prev_p1[1] - p1[1]))
        p2_diff = max(abs(prev_p2[0] - p2[0]), abs(prev_p2[1] - p2[1]))

        print(p1_diff)
        print(p2_diff)

        if p1_diff > MOVEMENT_THRESHOLD and p2_diff > MOVEMENT_THRESHOLD:
            IS_MOVING = True
        else:
            IS_MOVING = False

        if boxes[0, 4] > CONFIDENCE:
            SIGN_DETECTED = True
        else:
            SIGN_DETECTED = False

        prev_p1 = p1
        prev_p2 = p2

    counter += 1

    print(not IS_MOVING)
    print(SIGN_DETECTED)
    print(not driver1.is_disabled)
    if not IS_MOVING and SIGN_DETECTED and not driver1.is_disabled:
        print('Tukaj ne smete parkirati!')
        out_text = 'Tukaj ne smete parkirati!'
    else:
        out_text = ''


    print(IS_MOVING)
    print(counter)
    cv2.putText(whiteBackground, '', (30, 30), 1, 2, (0, 0, 255))
    cv2.putText(frame, str(IS_MOVING), (30, 30), 1, 2, (0, 0, 255))
    cv2.putText(whiteBackground, '', (1200, 30), 1, 2, (0, 0, 255))
    cv2.putText(frame, out_text, (500, 30), 1, 2, (0, 0, 255))

    plot_bboxes(frame, boxes, score=False, conf=CONFIDENCE)

    cv2.imshow('Neural Network Test Player', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    cv2.imshow('Neural Network Test Player', whiteBackground)

cap.release()
cv2.destroyAllWindows() # destroy all opened windows
