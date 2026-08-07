import cv2
import numpy as np

def is_actual_road_image(img_np: np.ndarray) -> tuple[bool, str]:
    """
    High-precision, offline 0-delay Road Scene Classifier.
    Analyzes color space, asphalt gray-scale ratio, texture density, and indoor vs outdoor lighting.
    """
    height, width = img_np.shape[:2]
    total_pixels = height * width
    
    # Convert HSV and Grayscale
    hsv = cv2.cvtColor(img_np, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
    
    h, s, v = cv2.split(hsv)
    
    # 1. Asphalt Neutrality Score: Asphalt roads are low saturation (s < 60) and medium brightness (30 < v < 200)
    asphalt_mask = (s < 65) & (v > 25) & (v < 210)
    asphalt_ratio = np.sum(asphalt_mask) / float(total_pixels)
    
    # 2. Indoor High-Saturation / Artificial Color check (furniture, clothes, toys, food, pets, screens)
    high_sat_mask = (s > 100) & (v > 50)
    high_sat_ratio = np.sum(high_sat_mask) / float(total_pixels)
    
    # 3. Very bright indoor wall/paper/screen check
    bright_flat_mask = (s < 20) & (v > 220)
    bright_flat_ratio = np.sum(bright_flat_mask) / float(total_pixels)
    
    # 4. Texture edge density in lower half (ground plane)
    lower_half_gray = gray[int(height * 0.4):, :]
    edges = cv2.Canny(lower_half_gray, 30, 120)
    edge_density = np.sum(edges > 0) / float(lower_half_gray.size)
    
    # Decision Rules:
    # If high saturation objects dominate (e.g. colorful room, clothes, toy, food, face, pet)
    if high_sat_ratio > 0.28:
        return False, "These are not road images, please upload actual road images."
        
    # If bright flat wall/paper/document dominates
    if bright_flat_ratio > 0.35:
        return False, "These are not road images, please upload actual road images."
        
    # Road surface ground ratio check
    if asphalt_ratio < 0.22 and edge_density < 0.015:
        return False, "These are not road images, please upload actual road images."
        
    return True, "Valid Road Scene"


if __name__ == "__main__":
    # Test sample road image
    pothole_img = cv2.imread("uploads/sample_pothole_demo.jpg")
    if pothole_img is not None:
        valid, msg = is_actual_road_image(pothole_img)
        print("Pothole demo test:", valid, "|", msg)
        
    # Test synthetic non-road image (high saturation colorful room)
    fake_room = np.zeros((400, 400, 3), dtype=np.uint8)
    fake_room[:, :] = (0, 0, 255) # Red box
    valid_fake, msg_fake = is_actual_road_image(fake_room)
    print("Fake non-road test:", valid_fake, "|", msg_fake)
