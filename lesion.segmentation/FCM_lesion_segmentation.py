import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2lab
from skfuzzy import cmeans, membership

def preprocess_image(image_path):
    # Read the image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Convert to LAB color space
    lab_img = rgb2lab(img)
    
    # Normalize the L channel
    l_channel = lab_img[:,:,0]
    l_channel = (l_channel - l_channel.min()) / (l_channel.max() - l_channel.min())
    
    return img, l_channel

def apply_fcm(image, n_clusters=2, m=2, error=0.005, max_iter=1000):
    # Reshape the image to a 2D array of pixels
    pixels = image.reshape((-1, 1))
    
    # Apply FCM clustering
    cntr, u, _, _, _, _, _ = cmeans(pixels.T, n_clusters, m, error, max_iter)
    
    # Get the segmentation mask
    segmentation = np.argmax(u, axis=0).reshape(image.shape)
    
    return segmentation

def post_process(segmentation):
    # Apply morphological operations to refine the segmentation
    kernel = np.ones((5,5), np.uint8)
    segmentation = cv2.morphologyEx(segmentation.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    segmentation = cv2.morphologyEx(segmentation, cv2.MORPH_OPEN, kernel)
    
    return segmentation

def visualize_results(original_img, segmentation):
    # Create a binary mask
    mask = (segmentation == 1).astype(np.uint8)
    
    # Apply the mask to the original image
    segmented_img = cv2.bitwise_and(original_img, original_img, mask=mask)
    
    # Visualize the results
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    ax1.imshow(original_img)
    ax1.set_title('Original Image')
    ax1.axis('off')
    
    ax2.imshow(segmentation, cmap='gray')
    ax2.set_title('Segmentation Mask')
    ax2.axis('off')
    
    ax3.imshow(segmented_img)
    ax3.set_title('Segmented Lesion')
    ax3.axis('off')
    
    plt.tight_layout()
    plt.show()

def main(image_path):
    # Preprocess the image
    original_img, preprocessed_img = preprocess_image(image_path)
    
    # Apply FCM
    segmentation = apply_fcm(preprocessed_img)
    
    # Post-process the segmentation
    refined_segmentation = post_process(segmentation)
    
    # Visualize the results
    visualize_results(original_img, refined_segmentation)

# Example usage
image_path='./lesion.images/image1.png'

main(image_path)