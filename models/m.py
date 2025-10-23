import torch
import ultralytics.nn.tasks as tasks

# YOLO model এর DetectionModel কে safe global list এ allow করা হচ্ছে
torch.serialization.add_safe_globals([tasks.DetectionModel])

# এখন CPU তে safe ভাবে load করা যাবে
ckpt = torch.load("yolov11_trained.pt", map_location="cpu", weights_only=False)

# CPU compatible model হিসেবে save করো
torch.save(ckpt, "yolov11_trained_cpu.pt")

print("✅ CPU-compatible YOLO model saved successfully!")