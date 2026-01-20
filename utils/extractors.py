"""
Multi-Modal Extraction Engines
Implements VLM, OCR, and CV-based extraction with ensemble fusion
"""

import re
import numpy as np
from typing import List, Dict, Any, Tuple
import cv2
from paddleocr import PaddleOCR
from ultralytics import YOLO
import warnings
warnings.filterwarnings('ignore')


class VLMExtractor:
    """Vision-Language Model based extractor using Qwen2.5-VL"""
    
    def __init__(self):
        """Initialize VLM model"""
        print("    Loading VLM (Qwen2.5-VL-2B)...")
        # In production, load actual model
        # from transformers import Qwen2VLForConditionalGeneration
        self.model_loaded = True
    
    def extract(self, images: List[np.ndarray], language: str) -> Dict[str, Any]:
        """
        Extract fields using VLM
        
        Args:
            images: List of document images
            language: Detected language
            
        Returns:
            Extracted fields with confidence scores
        """
        # Simulate VLM extraction
        # In production, this would call the actual model
        
        prompt = f"""
        Extract the following fields from this invoice document:
        1. Dealer Name
        2. Model Name (tractor or vehicle model)
        3. Horse Power (numeric value only)
        4. Asset Cost (total cost, numeric value only)
        5. Presence of Dealer Signature (yes/no with bounding box)
        6. Presence of Dealer Stamp (yes/no with bounding box)
        
        Return as JSON format with confidence scores.
        """
        
        # Mock response (replace with actual VLM inference)
        return {
            'dealer_name': {'value': 'ABC Tractors Pvt Ltd', 'confidence': 0.92},
            'model_name': {'value': 'Mahindra 575 DI', 'confidence': 0.95},
            'horse_power': {'value': 50, 'confidence': 0.88},
            'asset_cost': {'value': 525000, 'confidence': 0.90},
            'signature': {
                'present': True,
                'bbox': [100, 200, 300, 250],
                'confidence': 0.85
            },
            'stamp': {
                'present': True,
                'bbox': [400, 500, 500, 550],
                'confidence': 0.87
            }
        }


class OCRExtractor:
    """OCR-based extractor using PaddleOCR + Layout Analysis"""
    
    def __init__(self):
        """Initialize OCR engine"""
        print("    Loading OCR (PaddleOCR)...")
        self.ocr = PaddleOCR(
            use_angle_cls=True,
            lang='en',
            show_log=False
        )
    
    def extract(self, images: List[np.ndarray], language: str) -> Dict[str, Any]:
        """
        Extract fields using OCR + regex patterns
        
        Args:
            images: List of document images
            language: Detected language
            
        Returns:
            Extracted fields with confidence scores
        """
        all_text = []
        
        # Run OCR on all pages
        for image in images:
            result = self.ocr.ocr(image, cls=True)
            
            if result and result[0]:
                for line in result[0]:
                    text = line[1][0]
                    conf = line[1][1]
                    bbox = line[0]
                    all_text.append({
                        'text': text,
                        'confidence': conf,
                        'bbox': bbox
                    })
        
        # Extract fields using pattern matching
        extracted = self._extract_fields_from_text(all_text)
        
        return extracted
    
    def _extract_fields_from_text(self, text_blocks: List[Dict]) -> Dict[str, Any]:
        """Extract structured fields from OCR text"""
        
        # Combine all text
        full_text = ' '.join([t['text'] for t in text_blocks])
        
        # Patterns for field extraction
        patterns = {
            'dealer_name': r'(?:dealer|vendor|seller)[\s:]+([A-Za-z\s&.]+(?:Ltd|Pvt|Inc)?)',
            'model_name': r'(?:model|tractor|vehicle)[\s:]+([A-Za-z0-9\s]+(?:DI|HP)?)',
            'horse_power': r'(\d+)\s*(?:HP|hp|Horse\s*Power)',
            'asset_cost': r'(?:cost|price|amount|total)[\s:Rs.₹]*(\d+(?:,\d+)*)'
        }
        
        extracted = {}
        
        for field, pattern in patterns.items():
            match = re.search(pattern, full_text, re.IGNORECASE)
            if match:
                value = match.group(1).strip()
                
                # Clean numeric fields
                if field in ['horse_power', 'asset_cost']:
                    value = int(re.sub(r'[^\d]', '', value))
                
                extracted[field] = {
                    'value': value,
                    'confidence': 0.80
                }
            else:
                extracted[field] = {
                    'value': None,
                    'confidence': 0.0
                }
        
        # Signature and stamp detection (simplified)
        extracted['signature'] = {
            'present': False,
            'bbox': [0, 0, 0, 0],
            'confidence': 0.5
        }
        extracted['stamp'] = {
            'present': False,
            'bbox': [0, 0, 0, 0],
            'confidence': 0.5
        }
        
        return extracted


class CVExtractor:
    """Computer Vision extractor using YOLO for signature/stamp detection"""
    
    def __init__(self):
        """Initialize CV models"""
        print("    Loading CV models (YOLO)...")
        # In production, load trained YOLO model
        # self.model = YOLO('signature_stamp_detector.pt')
        self.model_loaded = True
    
    def extract(self, images: List[np.ndarray], language: str) -> Dict[str, Any]:
        """
        Extract visual elements (signatures, stamps)
        
        Args:
            images: List of document images
            language: Detected language
            
        Returns:
            Detected visual elements with bounding boxes
        """
        # Detect signatures and stamps using CV techniques
        signature_bbox = self._detect_signature(images[0])
        stamp_bbox = self._detect_stamp(images[0])
        
        return {
            'dealer_name': {'value': None, 'confidence': 0.0},
            'model_name': {'value': None, 'confidence': 0.0},
            'horse_power': {'value': None, 'confidence': 0.0},
            'asset_cost': {'value': None, 'confidence': 0.0},
            'signature': {
                'present': signature_bbox is not None,
                'bbox': signature_bbox if signature_bbox else [0, 0, 0, 0],
                'confidence': 0.85 if signature_bbox else 0.2
            },
            'stamp': {
                'present': stamp_bbox is not None,
                'bbox': stamp_bbox if stamp_bbox else [0, 0, 0, 0],
                'confidence': 0.82 if stamp_bbox else 0.2
            }
        }
    
    def _detect_signature(self, image: np.ndarray) -> List[int]:
        """Detect signature using contour analysis"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
            
            # Apply thresholding
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # Find contours
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter contours that could be signatures (handwriting-like)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if 1000 < area < 50000:  # Reasonable signature size
                    x, y, w, h = cv2.boundingRect(cnt)
                    aspect_ratio = w / float(h) if h > 0 else 0
                    
                    if 1.5 < aspect_ratio < 5.0:  # Signature aspect ratio
                        return [int(x), int(y), int(x+w), int(y+h)]
            
            return None
        except:
            return None
    
    def _detect_stamp(self, image: np.ndarray) -> List[int]:
        """Detect stamp using circular/rectangular detection"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
            
            # Detect circles (circular stamps)
            circles = cv2.HoughCircles(
                gray,
                cv2.HOUGH_GRADIENT,
                dp=1,
                minDist=50,
                param1=50,
                param2=30,
                minRadius=20,
                maxRadius=100
            )
            
            if circles is not None:
                circles = np.uint16(np.around(circles))
                for circle in circles[0, :1]:  # Take first circle
                    x, y, r = circle
                    return [int(x-r), int(y-r), int(x+r), int(y+r)]
            
            return None
        except:
            return None


class EnsembleExtractor:
    """Ensemble extractor combining VLM, OCR, and CV results"""
    
    def __init__(self, vlm: VLMExtractor, ocr: OCRExtractor, cv: CVExtractor):
        """Initialize ensemble"""
        self.vlm = vlm
        self.ocr = ocr
        self.cv = cv
        
        # Weights for ensemble voting
        self.weights = {
            'vlm': 0.5,
            'ocr': 0.35,
            'cv': 0.15
        }
    
    def extract_fast(self, images: List[np.ndarray], language: str) -> Dict[str, Any]:
        """Fast extraction using VLM only"""
        return self.vlm.extract(images, language)
    
    def extract_standard(self, images: List[np.ndarray], language: str) -> Dict[str, Any]:
        """Standard extraction using VLM + OCR"""
        vlm_results = self.vlm.extract(images, language)
        ocr_results = self.ocr.extract(images, language)
        
        return self._merge_results([vlm_results, ocr_results], ['vlm', 'ocr'])
    
    def extract_robust(self, images: List[np.ndarray], language: str) -> Dict[str, Any]:
        """Robust extraction using all extractors"""
        vlm_results = self.vlm.extract(images, language)
        ocr_results = self.ocr.extract(images, language)
        cv_results = self.cv.extract(images, language)
        
        return self._merge_results(
            [vlm_results, ocr_results, cv_results],
            ['vlm', 'ocr', 'cv']
        )
    
    def _merge_results(
        self,
        results: List[Dict[str, Any]],
        sources: List[str]
    ) -> Dict[str, Any]:
        """Merge results from multiple extractors using weighted voting"""
        
        merged = {}
        text_fields = ['dealer_name', 'model_name', 'horse_power', 'asset_cost']
        visual_fields = ['signature', 'stamp']
        
        # Merge text fields using confidence-weighted selection
        for field in text_fields:
            candidates = []
            for result, source in zip(results, sources):
                if field in result and result[field]['value'] is not None:
                    weighted_conf = result[field]['confidence'] * self.weights[source]
                    candidates.append((result[field]['value'], weighted_conf))
            
            if candidates:
                # Select candidate with highest weighted confidence
                best_value, best_conf = max(candidates, key=lambda x: x[1])
                merged[field] = {'value': best_value, 'confidence': best_conf}
            else:
                merged[field] = {'value': None, 'confidence': 0.0}
        
        # Merge visual fields using IoU and confidence
        for field in visual_fields:
            best_detection = None
            best_score = 0
            
            for result, source in zip(results, sources):
                if field in result and result[field]['present']:
                    score = result[field]['confidence'] * self.weights[source]
                    if score > best_score:
                        best_score = score
                        best_detection = result[field]
            
            if best_detection:
                merged[field] = best_detection
            else:
                merged[field] = {
                    'present': False,
                    'bbox': [0, 0, 0, 0],
                    'confidence': 0.0
                }
        
        return merged