import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import slic, mark_boundaries
from sklearn.cluster import KMeans

class RoadExtractor:
    def __init__(self, image_path, scale_factor=1.0):
        """
        Initialize the RoadExtractor with an image and an optional scale factor.
        
        Parameters:
            image_path (str): Path to the image.
            scale_factor (float): Factor by which to scale the image for processing.
                                  A value < 1.0 will process a smaller version.
        """
        # Load the original image.
        self.original_image = cv2.imread(image_path)
        if self.original_image is None:
            raise ValueError(f"Image not found at {image_path}")
        
        self.scale_factor = scale_factor
        
        # If scaling is requested, resize the image.
        if scale_factor != 1.0:
            self.image = cv2.resize(self.original_image, (0, 0), fx=scale_factor, fy=scale_factor)
        else:
            self.image = self.original_image


    def extract_road_region(self, image, num_segments=100, k_clusters=3):
        """
        Extract the road region using superpixel segmentation and clustering.
        
        Parameters:
            image (numpy.ndarray): The input image (possibly scaled).
            num_segments (int): Number of superpixels for SLIC segmentation.
            k_clusters (int): Number of clusters for KMeans.
        
        Returns:
            tuple: (road_region, road_mask, segments)
        """
        # 1. Convert to LAB color space.
        lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

        # 2. Superpixel segmentation with reduced number of segments.
        segments = slic(lab_image, n_segments=num_segments, compactness=10, sigma=1, start_label=1)
        
        # 3. Compute features and vegetation flags for each superpixel.
        superpixel_ids = np.unique(segments)
        features = []
        vegetation_flags = []
        for seg_id in superpixel_ids:
            mask = segments == seg_id

            # Mean LAB color for the superpixel.
            mean_lab = lab_image[mask].mean(axis=0)

            # Centroid of the superpixel (row, col).
            coords = np.argwhere(mask)
            centroid = coords.mean(axis=0)
            feature_vector = np.hstack((mean_lab, centroid))
            features.append(feature_vector)
            
            # Compute average color in original image (BGR) for vegetation flagging.
            avg_bgr = self.image[mask].mean(axis=0)
            avg_bgr_uint8 = np.uint8([[avg_bgr]])
            avg_hsv = cv2.cvtColor(avg_bgr_uint8, cv2.COLOR_BGR2HSV)
            hue = avg_hsv[0, 0, 0]
            # Flag superpixel as vegetation if hue is within green range.
            vegetation_flags.append(35 <= hue <= 85)
                
        features = np.array(features)
        
        # 4. Normalize features (column-wise normalization).
        features_norm = (features - features.mean(axis=0)) / features.std(axis=0)
        
        # 5. Cluster the superpixels using KMeans.
        kmeans = KMeans(n_clusters=k_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(features_norm)

        cluster_map = np.zeros_like(segments, dtype=np.uint8)
        for seg_id, cl in zip(superpixel_ids, cluster_labels):
            cluster_map[segments == seg_id] = cl
        
        # 6. Identify the road cluster based on the highest average y-coordinate.
        cluster_y_means = []
        for i in range(k_clusters):
            y_values = features[cluster_labels == i, 3]
            cluster_y_means.append(np.mean(y_values))
        road_cluster = np.argmax(cluster_y_means)
        
        # 7. Create a binary road mask, excluding superpixels flagged as vegetation.
        road_mask = np.zeros(segments.shape, dtype=np.uint8)
        for idx, (seg_id, cluster) in enumerate(zip(superpixel_ids, cluster_labels)):
            if cluster == road_cluster and not vegetation_flags[idx]:
                road_mask[segments == seg_id] = 255
                
        # 8. Refine the mask with morphological closing.
        kernel = np.ones((5, 5), np.uint8)
        road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_CLOSE, kernel)
        
        # 9. Extract the road region from the image using the mask.
        road_region = cv2.bitwise_and(image, image, mask=road_mask)
        
        # 10. If processing was done on a scaled image, resize the outputs back to original dimensions.
        if self.scale_factor != 1.0:
            orig_shape = (self.original_image.shape[1], self.original_image.shape[0])
            road_mask = cv2.resize(road_mask, orig_shape, interpolation=cv2.INTER_NEAREST)
            road_region = cv2.resize(road_region, orig_shape, interpolation=cv2.INTER_LINEAR)
            cluster_map = cv2.resize(cluster_map, orig_shape, interpolation=cv2.INTER_NEAREST)
        
        return road_region, road_mask, segments, cluster_map

    def run(self, num_segments=100, k_clusters=3):
        """
        Run the road extraction pipeline.
        
        Parameters:
            num_segments (int): Number of segments for SLIC.
            k_clusters (int): Number of clusters for KMeans.
        
        Returns:
            tuple: (road_region, road_mask, segments)
        """
        road_region, road_mask, segments, cluster_map = self.extract_road_region(self.image, num_segments, k_clusters)
        return road_region, road_mask, segments, cluster_map
    
def main():
    road_extractor = RoadExtractor("test.jpg", scale_factor=0.5)
    
    road_region, road_mask, segments, cluster_map = road_extractor.run(num_segments=100, k_clusters=3)

    segments_boundaries = mark_boundaries(cv2.cvtColor(road_extractor.image, cv2.COLOR_BGR2RGB), segments)
    
    fig, ax = plt.subplots(1, 5, figsize=(25, 5))
    ax[0].imshow(cv2.cvtColor(road_extractor.original_image, cv2.COLOR_BGR2RGB))
    ax[0].set_title("Original Image")
    ax[0].axis('off')

    ax[1].imshow(cluster_map, cmap='jet')
    ax[1].set_title("Superpixel Clusters")
    ax[1].axis('off')

    ax[2].imshow(segments_boundaries)
    ax[2].set_title("Superpixel Segments")
    ax[2].axis('off')

    ax[3].imshow(road_mask, cmap='gray')
    ax[3].set_title("Road Mask")
    ax[3].axis('off')
    
    ax[4].imshow(cv2.cvtColor(road_region, cv2.COLOR_BGR2RGB))
    ax[4].set_title("Road Region")
    ax[4].axis('off')
    
    plt.show()

if __name__ == "__main__":  
    main()