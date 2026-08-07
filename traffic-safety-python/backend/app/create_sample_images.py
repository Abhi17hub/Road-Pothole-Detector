import os
import cv2
import numpy as np

uploads_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "uploads")
os.makedirs(uploads_dir, exist_ok=True)

# 1. Pothole Sample
img_pothole = np.full((500, 700, 3), (80, 85, 90), dtype=np.uint8)
# Add asphalt texture noise
noise = np.random.normal(0, 15, img_pothole.shape).astype(np.int16)
img_pothole = np.clip(img_pothole.astype(np.int16) + noise, 0, 255).astype(np.uint8)
# Draw dark deep pothole cavity
cv2.ellipse(img_pothole, (350, 280), (120, 80), 15, 0, 360, (25, 28, 32), -1)
cv2.ellipse(img_pothole, (340, 275), (100, 65), 15, 0, 360, (15, 18, 22), -1)
cv2.ellipse(img_pothole, (350, 280), (120, 80), 15, 0, 360, (40, 45, 50), 4)

# 2. Crack Sample
img_crack = np.full((500, 700, 3), (90, 95, 100), dtype=np.uint8)
noise = np.random.normal(0, 12, img_crack.shape).astype(np.int16)
img_crack = np.clip(img_crack.astype(np.int16) + noise, 0, 255).astype(np.uint8)
# Draw jagged crack lines
pts = np.array([[100, 400], [220, 310], [350, 290], [480, 200], [600, 120]], np.int32)
pts = pts.reshape((-1, 1, 2))
cv2.polylines(img_crack, [pts], False, (20, 22, 25), 5)
cv2.polylines(img_crack, [pts], False, (40, 45, 50), 2)

# 3. Clean Road Sample
img_clean = np.full((500, 700, 3), (85, 90, 95), dtype=np.uint8)
noise = np.random.normal(0, 8, img_clean.shape).astype(np.int16)
img_clean = np.clip(img_clean.astype(np.int16) + noise, 0, 255).astype(np.uint8)
# White lane divider marking
cv2.rectangle(img_clean, (330, 0), (370, 500), (230, 235, 240), -1)

cv2.imwrite(os.path.join(uploads_dir, "sample_pothole_demo.jpg"), img_pothole)
cv2.imwrite(os.path.join(uploads_dir, "sample_crack_demo.jpg"), img_crack)
cv2.imwrite(os.path.join(uploads_dir, "sample_clean_demo.jpg"), img_clean)

print("Sample demo road images generated successfully in uploads directory.")
