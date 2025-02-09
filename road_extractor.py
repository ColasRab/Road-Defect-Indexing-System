import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import slic, mark_boundaries
from sklearn.cluster import KMeans

def apply_clahe(image, clipLimit=2.0, tileGridSize=(8, 8)):
    """
    Applies CLAHE to the input image.
    The image is first converted to LAB color space. CLAHE is applied to the L-channel,
    and then the image is converted back to BGR.
    """
    # Convert the image to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Create a CLAHE object and apply it to the L-channel
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    cl = clahe.apply(l)
    
    # Merge the enhanced L-channel with the original a and b channels
    lab_enhanced = cv2.merge((cl, a, b))
    
    # Convert back to BGR color space
    enhanced_image = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
    return enhanced_image

def increase_brightness(image, value=50):
    """
    Increase the brightness of the input image.

    Parameters:
      image (numpy.ndarray): Input image in BGR color space.
      value (int): The value to add to the V channel (default is 50). 
                   Increase for brighter image; lower for a subtler effect.

    Returns:
      numpy.ndarray: Brightness-enhanced image in BGR color space.
    """
    # Convert the image from BGR to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Split the HSV channels
    h, s, v = cv2.split(hsv)
    
    # Convert V channel to int16 to prevent overflow when adding the brightness value
    v = v.astype(np.int16)
    v += value  # Increase brightness
    v = np.clip(v, 0, 255)  # Ensure values stay within [0, 255]
    v = v.astype(np.uint8)
    
    # Merge the channels back and convert to BGR
    final_hsv = cv2.merge((h, s, v))
    bright_img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return bright_img

def extract_road_region(image, num_segments=200, k_clusters=3):
    """
    Extracts the road region from a scene image using superpixel segmentation and clustering.
    
    Steps:
      1. Convert the image to LAB color space.
      2. Segment the image into superpixels using SLIC.
      3. For each superpixel, compute a feature vector: [L, a, b, y, x],
         where (y, x) is the centroid (row, column) of the superpixel.
      4. Normalize these features.
      5. Cluster the superpixels using K‑means.
      6. Select the cluster whose superpixels have the highest average y-coordinate (i.e. located low in the image)
         as the road region.
      7. Create and return the road mask and the extracted road region.
    """
    # 1. Convert to LAB
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    lab_image = increase_brightness(lab_image, value=50)

    lab_image = apply_clahe(lab_image)
    
    # 2. Superpixel segmentation
    segments = slic(lab_image, n_segments=num_segments, compactness=10, sigma=1, start_label=1)
    
    # 3. Compute features for each superpixel
    superpixel_ids = np.unique(segments)
    features = []  # Will hold feature vectors: [L, a, b, y, x]
    for seg_id in superpixel_ids:
        mask = segments == seg_id
        # Mean LAB values for the superpixel
        mean_lab = lab_image[mask].mean(axis=0)  # [L, a, b]

        # Centroid of the superpixel: note that np.argwhere returns (row, col) = (y, x)
        coords = np.argwhere(mask)
        centroid = coords.mean(axis=0)  # [y, x]
        feature_vector = np.hstack((mean_lab, centroid))
        features.append(feature_vector)
    features = np.array(features)  # shape: (num_superpixels, 5)
    
    # 4. Normalize features (column-wise normalization)
    features_norm = (features - features.mean(axis=0)) / features.std(axis=0)
    
    # 5. Cluster the superpixels using K-means
    kmeans = KMeans(n_clusters=k_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(features_norm)
    
    # 6. Select the road cluster.
    # Assume the road is located in the lower part of the image.
    # (In our feature vector, the 4th element is y (vertical position).)
    cluster_y_means = []
    for i in range(k_clusters):
        # For cluster i, compute the mean y coordinate of its superpixels.
        y_values = features[cluster_labels == i, 3]
        cluster_y_means.append(np.mean(y_values))
    # The cluster with the highest average y is assumed to be the road.
    road_cluster = np.argmax(cluster_y_means)
    
    # 7. Create a binary mask from the selected cluster.
    road_mask = np.zeros(segments.shape, dtype=np.uint8)
    for seg_id, cluster in zip(superpixel_ids, cluster_labels):
        if cluster == road_cluster:
            road_mask[segments == seg_id] = 255
            
    # Optional: Clean up the mask using morphological operations.
    kernel = np.ones((5, 5), np.uint8)
    road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_CLOSE, kernel)
    
    # Extract the road region from the original image using the mask.
    road_region = cv2.bitwise_and(image, image, mask=road_mask)
    
    return road_region, road_mask, segments

def main():
    # Load the scene image (replace with your actual image path)
    image_path = 'datasets/road_defects/images/train/India_009655_jpg.rf.2353f9c2f527a96baaea1ecbae4ca7c2.jpg'

    image = cv2.imread(image_path)
    if image is None:
        print("Image not found at", image_path)
        return
    
    # Extract the road region
    road_region, road_mask, segments = extract_road_region(image)
    
    # Display results
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Original Scene Image")
    plt.axis("off")
    
    plt.subplot(2, 2, 2)
    plt.imshow(mark_boundaries(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), segments))
    plt.title("Superpixel Segmentation")
    plt.axis("off")
    
    plt.subplot(2, 2, 3)
    plt.imshow(road_mask, cmap='gray')
    plt.title("Road Mask")
    plt.axis("off")
    
    plt.subplot(2, 2, 4)
    plt.imshow(cv2.cvtColor(road_region, cv2.COLOR_BGR2RGB))
    plt.title("Extracted Road Region")
    plt.axis("off")
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
