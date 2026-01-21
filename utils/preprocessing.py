"""
Document Preprocessing Module
Handles image loading, enhancement, and quality assessment
"""

import cv2
import numpy as np
from PIL import Image
from typing import List, Dict, Any, Union
import fasttext
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class DocumentPreprocessor:
    """Preprocessing pipeline for invoice document images"""
    
    def __init__(self):
        """Initialize preprocessor with quality assessment model"""
        # Language detection model (lightweight)
        try:
            self.lang_model = fasttext.load_model('lid.176.bin')
        except:
            print("⚠️  Language detection model not found, using fallback")
            self.lang_model = None
    
    def process(self, image_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Main preprocessing pipeline for image files
        
        Args:
            image_path: Path to image file (PNG, JPG, etc.)
            
        Returns:
            Dictionary with processed images and metadata
        """
        # Load image
        images = self.load_images(image_path)
        
        if not images:
            raise ValueError(f"Failed to load image from {image_path}")
        
        # Assess quality
        quality_score = self.assess_quality(images[0])
        
        # Enhance images
        enhanced_images = [self.enhance_image(img) for img in images]
        
        # Detect language
        language = self.detect_language(enhanced_images[0])
        
        return {
            'images': enhanced_images,
            'quality_score': quality_score,
            'language': language,
            'num_pages': len(enhanced_images)
        }
    
    def load_images(self, image_path: Union[str, Path]) -> List[np.ndarray]:
        """
        Load image file(s)
        
        Args:
            image_path: Path to image file or directory
            
        Returns:
            List of images as numpy arrays
        """
        try:
            path = Path(image_path)
            
            # If it's a directory, load all images
            if path.is_dir():
                images = []
                valid_extensions = {'.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp'}
                
                for img_file in sorted(path.glob('*')):
                    if img_file.suffix.lower() in valid_extensions:
                        img = cv2.imread(str(img_file))
                        if img is not None:
                            # Convert BGR to RGB
                            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                            images.append(img)
                
                return images
            
            # Single image file
            elif path.is_file():
                img = cv2.imread(str(path))
                if img is not None:
                    # Convert BGR to RGB
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    return [img]
                else:
                    # Try with PIL as fallback
                    pil_img = Image.open(str(path))
                    img = np.array(pil_img.convert('RGB'))
                    return [img]
            
            return []
            
        except Exception as e:
            print(f"Error loading image: {e}")
            return []
    
    def assess_quality(self, image: np.ndarray) -> float:
        """
        Assess image quality using BRISQUE-inspired metrics
        
        Args:
            image: Input image
            
        Returns:
            Quality score between 0 and 1
        """
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
            
            # Calculate sharpness (Laplacian variance)
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            sharpness = laplacian.var()
            
            # Calculate contrast (standard deviation)
            contrast = gray.std()
            
            # Calculate brightness uniformity
            brightness = gray.mean()
            
            # Normalize scores (heuristic thresholds)
            sharpness_score = min(sharpness / 1000, 1.0)
            contrast_score = min(contrast / 100, 1.0)
            brightness_score = 1.0 - abs(brightness - 128) / 128
            
            # Weighted combination
            quality = (
                0.5 * sharpness_score +
                0.3 * contrast_score +
                0.2 * brightness_score
            )
            
            return float(quality)
        except:
            return 0.5  # Default medium quality
    
    def enhance_image(self, image: np.ndarray) -> np.ndarray:
        """
        Enhance image quality through preprocessing
        
        Args:
            image: Input image
            
        Returns:
            Enhanced image
        """
        try:
            # Convert to grayscale for processing
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image.copy()
            
            # Deskew
            gray = self.deskew(gray)
            
            # Denoise
            denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
            
            # Adaptive histogram equalization for better contrast
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(denoised)
            
            # Sharpen the image
            kernel = np.array([[-1, -1, -1],
                             [-1,  9, -1],
                             [-1, -1, -1]])
            enhanced = cv2.filter2D(enhanced, -1, kernel)
            
            # Convert back to RGB for consistency
            if len(image.shape) == 3:
                enhanced = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
            
            return enhanced
        except Exception as e:
            print(f"Error enhancing image: {e}")
            return image
    
    def deskew(self, image: np.ndarray) -> np.ndarray:
        """
        Deskew image using projection profile method
        
        Args:
            image: Input image
            
        Returns:
            Deskewed image
        """
        try:
            # Detect edges
            edges = cv2.Canny(image, 50, 150, apertureSize=3)
            
            # Detect lines using Hough transform
            lines = cv2.HoughLines(edges, 1, np.pi/180, 200)
            
            if lines is not None:
                # Calculate median angle
                angles = []
                for line in lines[:10]:  # Use top 10 lines
                    rho, theta = line[0]
                    angle = np.degrees(theta) - 90
                    if abs(angle) < 45:  # Ignore vertical lines
                        angles.append(angle)
                
                if angles:
                    median_angle = np.median(angles)
                    
                    # Only rotate if angle is significant (> 0.5 degrees)
                    if abs(median_angle) > 0.5:
                        # Rotate image
                        (h, w) = image.shape[:2]
                        center = (w // 2, h // 2)
                        M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
                        rotated = cv2.warpAffine(
                            image,
                            M,
                            (w, h),
                            flags=cv2.INTER_CUBIC,
                            borderMode=cv2.BORDER_REPLICATE
                        )
                        return rotated
            
            return image
        except Exception as e:
            print(f"Error deskewing image: {e}")
            return image
    
    def detect_language(self, image: np.ndarray) -> str:
        """
        Detect document language (simplified for images)
        
        Args:
            image: Input image
            
        Returns:
            Language code (e.g., 'en', 'hi', 'gu')
        """
        try:
            if self.lang_model is None:
                return "en"  # Default to English
            
            # For image-based detection, we would need OCR first
            # This is a simplified version - in production, 
            # extract text with OCR and then detect language
            
            # Check for common Indian scripts by analyzing Unicode ranges
            # after OCR (simplified here)
            
            return "en"  # Default to English for now
        except:
            return "en"
    
    def batch_process(self, image_paths: List[Union[str, Path]]) -> List[Dict[str, Any]]:
        """
        Process multiple images in batch
        
        Args:
            image_paths: List of paths to image files
            
        Returns:
            List of processed results
        """
        results = []
        
        for image_path in image_paths:
            try:
                result = self.process(image_path)
                result['image_path'] = str(image_path)
                results.append(result)
            except Exception as e:
                print(f"Failed to process {image_path}: {e}")
                continue
        
        return results