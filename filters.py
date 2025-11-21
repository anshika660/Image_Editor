import cv2
import numpy as np
import tkinter as tk
from tkinter import messagebox

class AdvancedFilters:
    def __init__(self):
        pass
    
    def grayscale(self, image):
        """Convert image to grayscale"""
        try:
            if len(image.shape) == 3:
                return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            return image
        except Exception as e:
            print(f"Error in grayscale conversion: {e}")
            return image
    
    def sepia(self, image):
        """Apply sepia filter"""
        try:
            kernel = np.array([[0.272, 0.534, 0.131],
                              [0.349, 0.686, 0.168],
                              [0.393, 0.769, 0.189]])
            sepia_image = cv2.transform(image, kernel)
            return np.clip(sepia_image, 0, 255).astype(np.uint8)
        except Exception as e:
            print(f"Error in sepia filter: {e}")
            return image
    
    def gaussian_blur(self, image, kernel_size=5):
        """Apply Gaussian blur"""
        try:
            # Ensure kernel size is odd
            if kernel_size % 2 == 0:
                kernel_size += 1
            return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
        except Exception as e:
            print(f"Error in Gaussian blur: {e}")
            return image
    
    def sharpen(self, image, kernel_size=3):
        """Apply sharpening filter"""
        try:
            kernel = np.array([[-1, -1, -1],
                              [-1,  9, -1],
                              [-1, -1, -1]])
            return cv2.filter2D(image, -1, kernel)
        except Exception as e:
            print(f"Error in sharpen filter: {e}")
            return image
    
    def edge_detection(self, image):
        """Apply Canny edge detection"""
        try:
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
                
            edges = cv2.Canny(gray, 100, 200)
            return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        except Exception as e:
            print(f"Error in edge detection: {e}")
            return image
    
    def emboss(self, image):
        """Apply emboss filter"""
        try:
            kernel = np.array([[-2, -1, 0],
                              [-1,  1, 1],
                              [ 0,  1, 2]])
            return cv2.filter2D(image, -1, kernel)
        except Exception as e:
            print(f"Error in emboss filter: {e}")
            return image
    
    def oil_painting(self, image, radius=3):
        """Apply oil painting effect"""
        try:
            # Simple implementation using bilateral filter
            return cv2.stylization(image, sigma_s=60, sigma_r=0.6)
        except Exception as e:
            print(f"Error in oil painting: {e}")
            return image
    
    def pencil_sketch(self, image):
        """Create pencil sketch effect"""
        try:
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
                
            inv_gray = 255 - gray
            blur = cv2.GaussianBlur(inv_gray, (21, 21), 0, 0)
            sketch = cv2.divide(gray, 255 - blur, scale=256)
            return cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)
        except Exception as e:
            print(f"Error in pencil sketch: {e}")
            return image
    
    def cartoon_effect(self, image):
        """Apply cartoon effect"""
        try:
            # Apply bilateral filter to reduce noise while keeping edges sharp
            color = cv2.bilateralFilter(image, 9, 300, 300)
            
            # Convert to grayscale and apply median blur
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            gray = cv2.medianBlur(gray, 7)
            
            # Detect edges using adaptive threshold
            edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                        cv2.THRESH_BINARY, 9, 2)
            
            # Convert edges back to color
            edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            
            # Combine color image with edges
            cartoon = cv2.bitwise_and(color, edges)
            return cartoon
        except Exception as e:
            print(f"Error in cartoon effect: {e}")
            return image
    
    def show_rgb_channels(self, image):
        """Display RGB channels separately"""
        try:
            if len(image.shape) == 3:
                channels = cv2.split(image)
                colors = ('Blue', 'Green', 'Red')
                
                for i, channel in enumerate(channels):
                    cv2.imshow(f'{colors[i]} Channel', channel)
                
                cv2.waitKey(0)
                cv2.destroyAllWindows()
        except Exception as e:
            print(f"Error showing RGB channels: {e}")
    
    def convert_to_hsv(self, image):
        """Convert image to HSV color space"""
        try:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)  # Convert back for display
        except Exception as e:
            print(f"Error in HSV conversion: {e}")
            return image
    
    def convert_to_lab(self, image):
        """Convert image to LAB color space"""
        try:
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)  # Convert back for display
        except Exception as e:
            print(f"Error in LAB conversion: {e}")
            return image
    
    def convert_to_yuv(self, image):
        """Convert image to YUV color space"""
        try:
            yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
            return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)  # Convert back for display
        except Exception as e:
            print(f"Error in YUV conversion: {e}")
            return image
    
    def face_detection(self, image):
        """Detect faces in the image"""
        try:
            # Load face cascade classifier
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
                
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            # Draw rectangles around faces
            result = image.copy()
            for (x, y, w, h) in faces:
                cv2.rectangle(result, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(result, 'Face', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 
                          0.5, (0, 255, 0), 2)
            
            messagebox.showinfo("Face Detection", f"Found {len(faces)} faces")
            return result
            
        except Exception as e:
            print(f"Error in face detection: {e}")
            messagebox.showerror("Error", f"Face detection failed: {e}")
            return image