"""
Document Preprocessing Module
Handles PDF conversion, image enhancement, and quality assessment
"""

import cv2
import numpy as np
from PIL import Image
import pdf2image
from typing import List, Dict, Any
import fasttext
from skimage import measure
import warnings
warnings.filterwarnings('ignore')


class DocumentPreprocessor:
    """Preprocessing pipeline for invoice documents"""
    
    def __init__(self):
        """Initialize preprocessor with quality assessment model"""
        # Language detection model (lightweight)
        try:
            self.lang_model = fasttext.load_model('lid.176.bin')
        except:
            print("⚠️  Language detection model not found, using fallback")
            self.lang_model = None
    
    def process(self, pdf_path: str) -> Dict[str, Any]:
        """
        Main preprocessing pipeline
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Dictionary with processed images and metadata
        """
        # Convert PDF to images
        images = self.pdf_to_images(pdf_path)
        
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
    
    def pdf_to_images(self, pdf_path: str, dpi: int = 300) -> List[np.ndarray]:
        """
        Convert PDF to high-resolution images
        
        Args:
            pdf_path: Path to PDF
            dpi: Resolution for conversion
            
        Returns:
            List of images as numpy arrays
        """
        try:
            pil_images = pdf2image.convert_from_path(
                pdf_path,
                dpi=dpi,
                fmt='png'
            )
            
            # Convert PIL to numpy arrays
            images = [np.array(img) for img in pil_images]
            
            return images
        except Exception as e:
            print(f"Error converting PDF: {e}")
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
            
            # Adaptive thresholding for better text clarity
            enhanced = cv2.adaptiveThreshold(
                denoised,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                11,
                2
            )
            
            # Convert back to RGB for consistency
            if len(image.shape) == 3:
                enhanced = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
            
            return enhanced
        except:
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
        except:
            return image
    
    def detect_language(self, image: np.ndarray) -> str:
        """
        Detect document language
        
        Args:
            image: Input image
            
        Returns:
            Language code (e.g., 'en', 'hi', 'gu')
        """
        try:
            if self.lang_model is None:
                return "en"  # Default to English
            
            # Simple heuristic: check for Unicode ranges
            # This is a fallback when fasttext is not available
            
            # For now, return English as default
            # In production, use OCR + fasttext on extracted text
            return "en"
        except:
            return "en"