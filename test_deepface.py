import os
import sys

# Minimal logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from deepface import DeepFace

test_img = r'F:\FaceTrain\Abella Danger\13827461_572180432987725_5032528573609541632_n.jpg'

print(f"=== Test Start ===")
print(f"Image exists: {os.path.exists(test_img)}")

try:
    print(f"Calling DeepFace.represent...")
    result = DeepFace.represent(
        img_path=test_img,
        model_name='ArcFace',
        detector_backend='retinaface',
        enforce_detection=True
    )
    print(f"SUCCESS!")
    print(f"Result type: {type(result)}")
    print(f"Result: {result}")
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()

print(f"=== Test End ===")