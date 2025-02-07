import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from skimage import restoration

def reduce_camera_vibration(image, kernel_size=(5,5)):
    """
    Reduce camera vibration effects using Gaussian blur and temporal stability
    """
    # Ensure image is in float32 format for better precision
    image_float = image.astype(np.float32) / 255.0
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(image_float, kernel_size, 0)
    # Convert back to uint8
    return (blurred * 255).astype(np.uint8)

def wiener_deblur(image, kernel_size=3, noise_level=0.01):
    """
    Apply Wiener deblurring filter with improved parameters
    """
    # Create motion blur kernel
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[int((kernel_size-1)/2), :] = np.ones(kernel_size)
    kernel = kernel / kernel_size
    
    # Convert to float32 for better precision
    image_float = image.astype(np.float32) / 255.0
    
    # Apply Wiener filter to each channel
    if len(image.shape) == 3:
        deblurred = np.zeros_like(image_float)
        for i in range(3):
            deblurred[:,:,i] = restoration.wiener(
                image_float[:,:,i], 
                kernel, 
                noise_level,
                clip=True  # Prevent overflow
            )
        # Ensure values are in [0, 1] range
        deblurred = np.clip(deblurred, 0, 1)
    else:
        deblurred = restoration.wiener(image_float, kernel, noise_level, clip=True)
        deblurred = np.clip(deblurred, 0, 1)
    
    return (deblurred * 255).astype(np.uint8)

def enhance_color(image, is_night=False):
    """
    Enhanced color correction with better parameters
    """
    # Convert to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    if is_night:
        # Night: Improved histogram equalization with adaptive gamma
        mean_brightness = np.mean(l)
        gamma = 1.5 if mean_brightness < 127 else 1.2
        l_gamma = np.power(l / 255.0, gamma) * 255.0
        l_eq = cv2.equalizeHist(l_gamma.astype(np.uint8))
    else:
        # Day: Improved CLAHE parameters
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        l_eq = clahe.apply(l)
    
    # Enhance color saturation slightly
    a = cv2.add(a, (a * 0.2).astype(np.uint8))
    b = cv2.add(b, (b * 0.2).astype(np.uint8))
    
    # Merge channels and convert back
    enhanced_lab = cv2.merge([l_eq, a, b])
    enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    return enhanced_bgr

def process_image(image_path, is_night=False):
    """
    Preprocessing pipeline optimized for road defect segmentation
    """
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    # Get image dimensions
    height, width = image.shape[:2]
    
    # Define ROI for road area
    roi_vertices = np.array([[
        (width * -0.2, height),
        (width * 0.2, height * 0.5),
        (width * 0.8, height * 0.5),
        (width * 1.2, height)
    ]], dtype=np.int32)
    
    # Create ROI mask
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, roi_vertices, (255, 255, 255))
    
    # Extract road area
    road_image = cv2.bitwise_and(image, mask)
    
    try:
        # 1. Stabilization and noise reduction
        stabilized = cv2.GaussianBlur(road_image, (5,5), 0)
        
        # 2. Enhance contrast for defect visibility
        lab = cv2.cvtColor(stabilized, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        l_eq = clahe.apply(l)
        enhanced_lab = cv2.merge([l_eq, a, b])
        enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        
        # 3. Convert to grayscale
        gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
        
        # 4. Edge detection for defect boundaries
        edges = cv2.Canny(gray, 50, 150)
        
        # 5. Defect segmentation
        # Apply local thresholding
        binary = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            11,
            2
        )
        
        # Combine edge detection with thresholding
        combined = cv2.bitwise_or(edges, binary)
        
        # 6. Clean up the segmentation
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
        # Remove noise
        morph = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel, iterations=1)
        # Connect nearby defects
        morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # 7. Final defect mask
        # Apply distance transform to get defect probability
        dist_transform = cv2.distanceTransform(morph, cv2.DIST_L2, 5)
        # Normalize to 0-255 range
        cv2.normalize(dist_transform, dist_transform, 0, 255, cv2.NORM_MINMAX)
        defect_mask = dist_transform.astype(np.uint8)
        
        # 8. Apply ROI mask to final result
        defect_mask = cv2.bitwise_and(defect_mask, mask[:,:,0])
        
    except Exception as e:
        print(f"Error in preprocessing: {str(e)}")
        raise
    
    return {
        'original': image,
        'road_roi': road_image,
        'stabilized': stabilized,
        'enhanced': enhanced,
        'gray': gray,
        'edges': edges,
        'binary': binary,
        'defect_mask': defect_mask
    }

def display_results(results):
    """
    Display preprocessing results with focus on defect segmentation
    """
    plt.figure(figsize=(15, 10))
    
    # Original
    plt.subplot(3, 3, 1)
    plt.imshow(cv2.cvtColor(results['original'], cv2.COLOR_BGR2RGB))
    plt.title("Original Image")
    plt.axis('off')
    
    # ROI
    plt.subplot(3, 3, 2)
    plt.imshow(cv2.cvtColor(results['road_roi'], cv2.COLOR_BGR2RGB))
    plt.title("Road ROI")
    plt.axis('off')
    
    # Enhanced
    plt.subplot(3, 3, 3)
    plt.imshow(cv2.cvtColor(results['enhanced'], cv2.COLOR_BGR2RGB))
    plt.title("Enhanced")
    plt.axis('off')
    
    # Grayscale
    plt.subplot(3, 3, 4)
    plt.imshow(results['gray'], cmap='gray')
    plt.title("Grayscale")
    plt.axis('off')
    
    # Edges
    plt.subplot(3, 3, 5)
    plt.imshow(results['edges'], cmap='gray')
    plt.title("Edge Detection")
    plt.axis('off')
    
    # Binary
    plt.subplot(3, 3, 6)
    plt.imshow(results['binary'], cmap='gray')
    plt.title("Binary Threshold")
    plt.axis('off')
    
    # Defect Mask
    plt.subplot(3, 3, 7)
    plt.imshow(results['defect_mask'], cmap='jet')
    plt.title("Defect Mask")
    plt.axis('off')
    
    # Overlay
    overlay = results['original'].copy()
    mask_rgb = cv2.cvtColor(results['defect_mask'], cv2.COLOR_GRAY2BGR)
    overlay = cv2.addWeighted(overlay, 0.7, mask_rgb, 0.3, 0)
    plt.subplot(3, 3, 8)
    plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    plt.title("Overlay")
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    image_path = "datasets/road_defects/images/train/India_000027_jpg.rf.753f9f55480be9a48b17d149f1a08800.jpg"
    results = process_image(image_path, is_night=False)
    display_results(results)