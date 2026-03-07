import cv2
import numpy as np

def get_drawing_points(image_path):
    # 1. Load the image
    img = cv2.imread(image_path, 0) # Load as grayscale
    
    # 2. Resize to A4 proportions (keeping it simple)
    # Let's say 210 pixels wide, 297 pixels high
    img_resized = cv2.resize(img, (210, 297))
    
    # 3. Canny Edge Detection (This finds the lines)
    edges = cv2.Canny(img_resized, 100, 200)
    
    # 4. Get coordinates of all white pixels (the edges)
    # np.column_stack returns (y, x), so we flip it to (x, y)
    points = np.column_stack(np.where(edges > 0))
    
    return points, edges

# Test it locally
if __name__ == "__main__":
    # Put a simple logo.png in your /images folder first!
    pts, edge_view = get_drawing_points('images/Testlogo.jpeg')
    print(f"Found {len(pts)} points to draw.")
    cv2.imshow("What the AI sees", edge_view)
    cv2.waitKey(0)