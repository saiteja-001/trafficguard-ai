import time
import numpy as np

# Model training metrics from evaluation results
TRAINED_MODEL_METRICS = {
    "overall": {
        "mAP50": 0.9487,
        "mAP50_95": 0.684,
        "precision": 0.877,
        "recall": 0.936,
        "epochs": 100,
        "batch_size": 16,
        "optimizer": "AdamW"
    },
    "classes": {
        "motorcyclist": {
            "precision": 0.912,
            "recall": 0.948,
            "mAP50": 0.959
        },
        "helmet": {
            "precision": 0.895,
            "recall": 0.931,
            "mAP50": 0.942
        },
        "license_plate": {
            "precision": 0.824,
            "recall": 0.929,
            "mAP50": 0.945
        }
    }
}

# General device performance metrics for comparison
HARDWARE_BENCHMARKS = {
    "YOLOv8s": {
        "CPU (Intel i7)": {"Inference": 85.2, "Preprocess": 2.1, "Postprocess": 1.5, "Total": 88.8},
        "GPU (NVIDIA RTX 4060)": {"Inference": 8.4, "Preprocess": 0.8, "Postprocess": 0.7, "Total": 9.9},
        "Edge Device (Jetson Orin Nano)": {"Inference": 18.5, "Preprocess": 1.2, "Postprocess": 1.1, "Total": 20.8}
    },
    "YOLOv8m": {
        "CPU (Intel i7)": {"Inference": 145.7, "Preprocess": 2.2, "Postprocess": 1.8, "Total": 149.7},
        "GPU (NVIDIA RTX 4060)": {"Inference": 12.1, "Preprocess": 0.9, "Postprocess": 0.8, "Total": 13.8},
        "Edge Device (Jetson Orin Nano)": {"Inference": 28.4, "Preprocess": 1.4, "Postprocess": 1.3, "Total": 31.1}
    }
}

def calculate_latencies(start_time, prep_end, det_end, ocr_end, db_end):
    """
    Calculates execution duration and percentages for each pipeline phase.
    """
    t_prep = (prep_end - start_time) * 1000 # ms
    t_det = (det_end - prep_end) * 1000 # ms
    t_ocr = (ocr_end - det_end) * 1000 # ms
    t_db = (db_end - ocr_end) * 1000 # ms
    t_total = t_prep + t_det + t_ocr + t_db
    
    breakdown = [
        {"Phase": "1. Preprocessing", "Time (ms)": round(t_prep, 1), "Share": round((t_prep/t_total)*100, 1)},
        {"Phase": "2. Object Detection", "Time (ms)": round(t_det, 1), "Share": round((t_det/t_total)*100, 1)},
        {"Phase": "3. License Plate OCR", "Time (ms)": round(t_ocr, 1), "Share": round((t_ocr/t_total)*100, 1)},
        {"Phase": "4. DB & PDF Logging", "Time (ms)": round(t_db, 1), "Share": round((t_db/t_total)*100, 1)},
    ]
    
    return {
        "total_ms": round(t_total, 1),
        "phases": breakdown
    }
