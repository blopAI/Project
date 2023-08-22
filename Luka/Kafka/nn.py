from PIL import Image
from ultralytics import YOLO
import pandas as pd
from ultralytics.yolo.utils.plotting import Annotator
import cv2

def load_model(path):
    model = YOLO(path)
    return model

def perform_inference(model, img_path):
    img = Image.open(img_path)
    results = model.predict(img)
    return results

def print_results(results):
    #print("RESULT TYPE: ",type(results))
    #print("RESULT: ", results)
    for result in results:
        boxes = result.boxes  
        masks = result.masks  
        probs = result.probs
        print(boxes, masks, probs)

def save_image(results, path):
    for i, result in enumerate(results):
        img = cv2.cvtColor(result.orig_img, cv2.COLOR_RGB2BGR)
        annotator = Annotator(img)
        for box in result.boxes.data:
            bbox = box[:4]  # get box coordinates
            cls = box[5]  # get class id
            annotator.box_label(bbox, result.names[int(cls)])
        img_with_boxes = annotator.result()
        img_path = f"{path}/image_{i}.jpg"
        cv2.imwrite(img_path, img_with_boxes)

if __name__ == '__main__':
    model_path = '/home/lukaknez/runs/detect/train21/weights/best.pt' 
    img_path = '/home/lukaknez/Desktop/data/images/train/Image26.jpg'  
    save_path = '/home/lukaknez/Desktop/neural_network/Results'  

    # Load model
    model = load_model(model_path)

    # Perform inference
    results = perform_inference(model, img_path)

    # Print results
    print_results(results)

    #Save output image
    save_image(results, save_path)
