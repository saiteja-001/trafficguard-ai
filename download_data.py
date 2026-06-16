from roboflow import Roboflow

rf = Roboflow(api_key="PdjqEqnLJdtCYchMzusj")

project = rf.workspace("nckh-2023").project("helmet-detection-project")
version = project.version(1)
dataset = version.download("yolov8", location="datasets/helmet")

print("Done!")