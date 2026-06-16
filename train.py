from ultralytics import YOLO

model = YOLO('yolov8m.pt')  

model.train(
    data='datasets/helmet/data.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    name='helmet_model',
    project='models',
    patience=20,
    save=True,
    optimizer='AdamW',
    lr0=0.001,
    lrf=0.01,
    momentum=0.937,
    weight_decay=0.0005,
    warmup_epochs=5,
    mosaic=1.0,
    mixup=0.1,
    flipud=0.1,
    fliplr=0.5,
    degrees=10.0,
    translate=0.1,
    scale=0.5,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    copy_paste=0.1,
    augment=True,
    val=True,
    plots=True,
    verbose=True
)

print("Training done!")