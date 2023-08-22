from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n.yaml")  # build a new model from scratch

# Use the model
model.train(data="config2.yaml", epochs=20, batch=4, device='cpu')  # train the model
