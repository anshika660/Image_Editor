import cv2
import numpy as np

class ImageProcessor:
    def __init__(self):
        pass
    
    def adjust_brightness_contrast(self, image, brightness=0, contrast=1.0):
        """Adjust brightness and contrast of the image"""
        try:
            # Convert to float for calculations
            image_float = image.astype(np.float32)
            
            # Apply brightness
            if brightness != 0:
                image_float += brightness
            
            # Apply contrast
            if contrast != 1.0:
                image_float *= contrast
            
            # Clip values to [0, 255] and convert back to uint8
            result = np.clip(image_float, 0, 255).astype(np.uint8)
            return result
            
        except Exception as e:
            print(f"Error in brightness/contrast adjustment: {e}")
            return image
    
    def adjust_saturation(self, image, saturation=1.0):
        """Adjust saturation of the image"""
        try:
            # Convert to HSV color space
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
            
            # Adjust saturation channel
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation, 0, 255)
            
            # Convert back to BGR
            result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
            return result
            
        except Exception as e:
            print(f"Error in saturation adjustment: {e}")
            return image
    
    def rotate_image(self, image, angle):
        """Rotate image by given angle"""
        try:
            height, width = image.shape[:2]
            center = (width // 2, height // 2)
            
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(image, rotation_matrix, (width, height), 
                                   flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
            return rotated
            
        except Exception as e:
            print(f"Error in rotation: {e}")
            return image
    
    def resize_image(self, image, width, height):
        """Resize image to specified dimensions"""
        try:
            return cv2.resize(image, (width, height), interpolation=cv2.INTER_CUBIC)
        except Exception as e:
            print(f"Error in resizing: {e}")
            return image
    
    def crop_image(self, image, x, y, width, height):
        """Crop image from (x,y) to (x+width, y+height)"""
        try:
            return image[y:y+height, x:x+width]
        except Exception as e:
            print(f"Error in cropping: {e}")
            return image
    
    def flip_image(self, image, flip_code):
        """Flip image horizontally (1), vertically (0), or both (-1)"""
        try:
            return cv2.flip(image, flip_code)
        except Exception as e:
            print(f"Error in flipping: {e}")
            return image
    
    def adjust_hue(self, image, hue_shift):
        """Adjust hue of the image"""
        try:
            # Convert to HSV
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
            
            # Adjust hue (wrapping around 180)
            hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift) % 180
            
            # Convert back to BGR
            result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
            return result
            
        except Exception as e:
            print(f"Error in hue adjustment: {e}")
            return image