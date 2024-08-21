from ultralytics import YOLO

# Load an official or custom model
model = YOLO("container.pt")  # Load an official Detect model


# Perform tracking with the model
results = model.track("https://youtu.be/pBASSCcczOI", show=True,save=True)  # Tracking with default tracker
