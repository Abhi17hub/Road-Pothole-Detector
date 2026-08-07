import cv2
import numpy as np

def is_actual_road_image(img_np: np.ndarray) -> tuple[bool, str]:
    height, width = img_np.shape[:2]
    total_pixels = height * width
    
    hsv = cv2.cvtColor(img_np, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
    
    h, s, v = cv2.split(hsv)
    
    # 1. Human Skin Tone Mask (HSV)
    skin_mask1 = (h >= 0) & (h <= 25) & (s >= 25) & (s <= 170) & (v >= 60) & (v <= 255)
    skin_mask2 = (h >= 170) & (h <= 180) & (s >= 25) & (s <= 170) & (v >= 60) & (v <= 255)
    skin_ratio = np.sum(skin_mask1 | skin_mask2) / float(total_pixels)
    
    # 2. White Paper / Document / Poster Background Mask
    paper_mask = (s < 25) & (v > 210)
    paper_ratio = np.sum(paper_mask) / float(total_pixels)
    
    # 3. High-Frequency Text Edge Density (Documents, Posters, Ads)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    text_edges = cv2.Canny(blur, 100, 200)
    text_edge_ratio = np.sum(text_edges > 0) / float(total_pixels)
    
    # 4. Lower-Half Road Asphalt Neutrality (s < 60, 30 < v < 200)
    lower_hsv = hsv[int(height * 0.4):, :]
    lower_s = lower_hsv[:, :, 1]
    lower_v = lower_hsv[:, :, 2]
    asphalt_mask_lower = (lower_s < 65) & (lower_v > 25) & (lower_v < 210)
    asphalt_ratio_lower = np.sum(asphalt_mask_lower) / float(lower_hsv.size)
    
    # Check 1: Human skin tone detected (person/portrait/indoor shop photo)
    if skin_ratio > 0.04 and asphalt_ratio_lower < 0.25:
        return False, "Person or indoor human photo detected! These are not road images, please upload actual road images."
        
    # Check 2: High document/poster paper content with text edges
    if paper_ratio > 0.22 and text_edge_ratio > 0.035 and asphalt_ratio_lower < 0.20:
        return False, "Printed document, poster or advertisement image detected! These are not road images, please upload actual road images."
        
    # Check 3: General low asphalt ratio in lower ground plane
    if asphalt_ratio_lower < 0.15:
        return False, "These are not road images, please upload actual road images."
        
    return True, "Valid Road Scene"

if __name__ == "__main__":
    pothole_img = cv2.imread("uploads/sample_pothole_demo.jpg")
    if pothole_img is not None:
        v, m = is_actual_road_image(pothole_img)
        print("Pothole image:", v, "|", m)
