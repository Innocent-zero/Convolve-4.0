"""
Teacher Extractors (VLM, OCR, CV)
These are used ONLY for generating pseudo-labels during training.
They are NOT used during inference with the trained SGAN model.
"""

import re
import numpy as np
from typing import List, Dict, Any, Optional
import cv2
from paddleocr import PaddleOCR
from PIL import Image
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
import warnings
import json
warnings.filterwarnings('ignore')
import cv2

class VLMExtractor:
    """Vision-Language Model (Teacher for pseudo-label generation)"""
    
    def __init__(self, model_name: str = "Qwen/Qwen2-VL-2B-Instruct"):
        print(f"    Loading Teacher VLM ({model_name})...")
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None
        )
        self.processor = AutoProcessor.from_pretrained(model_name)
        
        if self.device == "cpu":
            self.model = self.model.to(self.device)
        
        self.model.eval()
        print(f"    Teacher VLM loaded on {self.device}")
    
    def extract(self, images: List[np.ndarray], language: str) -> Dict[str, Any]:
        """Extract fields using VLM (robust version with debug)"""
        try:
            if not images:
                print("    VLM extraction: No images provided.")
                return self._get_empty()

            # Convert images to PIL
            pil_images = []
            for idx, img in enumerate(images):
                if img is None:
                    print(f"    VLM extraction: Image {idx} is None, skipping.")
                    continue
                try:
                    if img.dtype == np.uint8:
                        pil_images.append(Image.fromarray(img))
                    else:
                        pil_images.append(Image.fromarray((img * 255).astype(np.uint8)))
                except Exception as e:
                    print(f"    VLM extraction: Failed to convert image {idx} -> PIL: {e}")

            if not pil_images:
                print("    VLM extraction: No valid PIL images.")
                return self._get_empty()

            # Prepare the prompt
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {
                            "type": "text",
                            "text": (
                                "Extract the following from this invoice/quotation in JSON format:\n"
                                "{\n"
                                '  "dealer_name": {"value": "...", "confidence": 0.0-1.0},\n'
                                '  "model_name": {"value": "...", "confidence": 0.0-1.0},\n'
                                '  "horse_power": {"value": number, "confidence": 0.0-1.0},\n'
                                '  "asset_cost": {"value": number, "confidence": 0.0-1.0},\n'
                                '  "signature": {"present": true/false, "bbox": [x1, y1, x2, y2], "confidence": 0.0-1.0},\n'
                                '  "stamp": {"present": true/false, "bbox": [x1, y1, x2, y2], "confidence": 0.0-1.0}\n'
                                "}"
                            )
                        }
                    ]
                }
            ]

            prompt = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            if prompt is None:
                print("    VLM extraction: apply_chat_template returned None.")
                return self._get_empty()

            # Process inputs
            try:
                inputs = self.processor(
                    text=[prompt],
                    images=pil_images,
                    return_tensors="pt",
                    padding=True
                )
            except Exception as e:
                print(f"    VLM extraction: processor failed -> {e}")
                return self._get_empty()

            if inputs is None or "input_ids" not in inputs or "pixel_values" not in inputs:
                print("    VLM extraction: processor returned invalid inputs.")
                return self._get_empty()

            inputs = inputs.to(self.device)
            inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])

            # Generate output
            try:
                with torch.no_grad():
                    output_ids = self.model.generate(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        pixel_values=inputs["pixel_values"],
                        max_new_tokens=512
                    )
            except Exception as e:
                print(f"    VLM extraction: model.generate failed -> {e}")
                return self._get_empty()

            if output_ids is None:
                print("    VLM extraction: model.generate returned None.")
                return self._get_empty()

            # Decode
            try:
                generated_text = self.processor.batch_decode(output_ids, skip_special_tokens=True)[0]
            except Exception as e:
                print(f"    VLM extraction: batch_decode failed -> {e}")
                return self._get_empty()

            if not generated_text:
                print("    VLM extraction: generated_text is empty.")
                return self._get_empty()

            # Parse JSON
            return self._parse_response(generated_text)

        except Exception as e:
            print(f"    VLM extraction unexpected error: {e}")
            return self._get_empty()

    
    def _parse_response(self, response: str) -> Dict[str, Any]:
        try:
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                parsed = json.loads(json_match.group(0))
                return self._normalize(parsed)
            else:
                return self._get_empty()
        except:
            return self._get_empty()
    
    def _normalize(self, data: Dict) -> Dict[str, Any]:
        normalized = {}
        for field in ['dealer_name', 'model_name', 'horse_power', 'asset_cost']:
            if field in data:
                normalized[field] = {
                    'value': data[field].get('value'),
                    'confidence': float(data[field].get('confidence', 0.7))
                }
            else:
                normalized[field] = {'value': None, 'confidence': 0.0}
        
        for field in ['signature', 'stamp']:
            if field in data:
                normalized[field] = {
                    'present': bool(data[field].get('present', False)),
                    'bbox': data[field].get('bbox', [0, 0, 0, 0]),
                    'confidence': float(data[field].get('confidence', 0.5))
                }
            else:
                normalized[field] = {'present': False, 'bbox': [0, 0, 0, 0], 'confidence': 0.0}
        
        return normalized
    
    def _get_empty(self) -> Dict[str, Any]:
        return {
            'dealer_name': {'value': None, 'confidence': 0.0},
            'model_name': {'value': None, 'confidence': 0.0},
            'horse_power': {'value': None, 'confidence': 0.0},
            'asset_cost': {'value': None, 'confidence': 0.0},
            'signature': {'present': False, 'bbox': [0, 0, 0, 0], 'confidence': 0.0},
            'stamp': {'present': False, 'bbox': [0, 0, 0, 0], 'confidence': 0.0}
        }


class OCRExtractor:
    """OCR-based extractor (Teacher for pseudo-label generation)"""
    
    def __init__(self):
        print("    Loading Teacher OCR (PaddleOCR)...")
        self.ocr = PaddleOCR(use_angle_cls=False, lang='en')
    
    def extract(self, images: List[np.ndarray], language: str) -> Dict[str, Any]:
        """Extract fields using OCR + regex"""
        try:
            if not images:
                return self._get_empty()
            
            all_text = []
            
            for image in images:
                if image is None:
                    continue
                
                try:
                    result = self.ocr.ocr(image)
                    if not result or result[0] is None:
                        continue
                    for line in result[0]:
                        if line is None or len(line) < 2:
                            continue
                        all_text.append({
                            'text': line[1][0],
                            'confidence': float(line[1][1]),
                            'bbox': line[0]
                        })
                except Exception as e:
                    print(f"    OCR error on image: {e}")
                    continue

            return self._extract_fields(all_text)
        
        except Exception as e:
            print(f"    OCR extraction error: {e}")
            return self._get_empty()
    
    def _extract_fields(self, text_blocks: List[Dict]) -> Dict[str, Any]:
        full_text = ' '.join([t['text'] for t in text_blocks])
        
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
                if field in ['horse_power', 'asset_cost']:
                    value = int(re.sub(r'[^\d]', '', value))
                extracted[field] = {'value': value, 'confidence': 0.80}
            else:
                extracted[field] = {'value': None, 'confidence': 0.0}
        
        extracted['signature'] = {'present': False, 'bbox': [0, 0, 0, 0], 'confidence': 0.5}
        extracted['stamp'] = {'present': False, 'bbox': [0, 0, 0, 0], 'confidence': 0.5}
        
        return extracted
    
    def _get_empty(self) -> Dict[str, Any]:
        return {
            'dealer_name': {'value': None, 'confidence': 0.0},
            'model_name': {'value': None, 'confidence': 0.0},
            'horse_power': {'value': None, 'confidence': 0.0},
            'asset_cost': {'value': None, 'confidence': 0.0},
            'signature': {'present': False, 'bbox': [0, 0, 0, 0], 'confidence': 0.0},
            'stamp': {'present': False, 'bbox': [0, 0, 0, 0], 'confidence': 0.0}
        }


class CVExtractor:
    """Computer Vision extractor (Teacher for pseudo-label generation)"""
    
    def __init__(self, yolo_model_path: Optional[str] = None):
        print("    Loading Teacher CV models...")
        self.use_yolo = yolo_model_path is not None
        
        if self.use_yolo:
            try:
                from ultralytics import YOLO
                self.yolo_model = YOLO(yolo_model_path)
                print(f"    Loaded YOLO from {yolo_model_path}")
            except Exception as e:
                print(f"    Warning: Could not load YOLO: {e}")
                self.use_yolo = False
    
    def extract(self, images: List[np.ndarray], language: str) -> Dict[str, Any]:
        """Extract visual elements"""
        try:
            if not images or images[0] is None:
                return self._get_empty()
            
            if self.use_yolo:
                signature_result = self._detect_with_yolo(images[0], 'signature')
                stamp_result = self._detect_with_yolo(images[0], 'stamp')
            else:
                signature_result = self._detect_signature_traditional(images[0])
                stamp_result = self._detect_stamp_traditional(images[0])
            
            return {
                'dealer_name': {'value': None, 'confidence': 0.0},
                'model_name': {'value': None, 'confidence': 0.0},
                'horse_power': {'value': None, 'confidence': 0.0},
                'asset_cost': {'value': None, 'confidence': 0.0},
                'signature': signature_result,
                'stamp': stamp_result
            }
        except Exception as e:
            print(f"    CV extraction error: {e}")
            return self._get_empty()
    
    def _detect_signature_traditional(self, image: np.ndarray) -> Dict[str, Any]:
        """Detect signature using traditional CV"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()
            binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            morph = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
            contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            best_signature = None
            best_score = 0
            
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if 500 < area < 100000:
                    x, y, w, h = cv2.boundingRect(cnt)
                    aspect_ratio = w / float(h) if h > 0 else 0
                    
                    if 1.2 < aspect_ratio < 6.0 and h > 20:
                        roi = binary[y:y+h, x:x+w]
                        density = np.sum(roi > 0) / (w * h) if (w * h) > 0 else 0
                        
                        if 0.05 < density < 0.4:
                            perimeter = cv2.arcLength(cnt, True)
                            circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
                            irregularity = 1 - circularity
                            
                            score = 0.3 * (1 - abs(aspect_ratio - 3.0) / 3.0) + 0.3 * (1 - abs(density - 0.15) / 0.15) + 0.4 * irregularity
                            
                            if score > best_score and score > 0.4:
                                best_score = score
                                best_signature = [int(x), int(y), int(x+w), int(y+h)]
            
            if best_signature:
                return {'present': True, 'bbox': best_signature, 'confidence': min(0.95, 0.5 + best_score * 0.5)}
            else:
                return {'present': False, 'bbox': [0, 0, 0, 0], 'confidence': 0.2}
                
        except Exception as e:
            return {'present': False, 'bbox': [0, 0, 0, 0], 'confidence': 0.0}
    
    def _detect_stamp_traditional(self, image: np.ndarray) -> Dict[str, Any]:
        """Detect stamp using traditional CV"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()
            blurred = cv2.GaussianBlur(gray, (9, 9), 2)
            circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=100, param1=50, param2=30, minRadius=30, maxRadius=150)
            
            best_circle_score = 0
            best_circle = None
            
            if circles is not None:
                circles = np.uint16(np.around(circles))
                for circle in circles[0]:
                    x, y, r = circle
                    x1, y1 = max(0, x-r), max(0, y-r)
                    x2, y2 = min(gray.shape[1], x+r), min(gray.shape[0], y+r)
                    roi = gray[y1:y2, x1:x2]
                    edges = cv2.Canny(roi, 50, 150)
                    edge_density = np.sum(edges > 0) / edges.size if edges.size > 0 else 0
                    
                    if edge_density > 0.1:
                        if edge_density > best_circle_score:
                            best_circle_score = edge_density
                            best_circle = [int(x-r), int(y-r), int(x+r), int(y+r)]
            
            if best_circle and best_circle_score > 0.3:
                return {'present': True, 'bbox': best_circle, 'confidence': min(0.92, 0.5 + best_circle_score * 0.5)}
            else:
                return {'present': False, 'bbox': [0, 0, 0, 0], 'confidence': 0.2}
                
        except Exception as e:
            return {'present': False, 'bbox': [0, 0, 0, 0], 'confidence': 0.0}
    
    def _detect_with_yolo(self, image: np.ndarray, target_class: str) -> Dict[str, Any]:
        """Detect using YOLO"""
        try:
            results = self.yolo_model(image, verbose=False)
            for result in results:
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    if self.yolo_model.names[cls_id].lower() == target_class.lower():
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        return {'present': True, 'bbox': [int(x1), int(y1), int(x2), int(y2)], 'confidence': float(box.conf[0])}
            return {'present': False, 'bbox': [0, 0, 0, 0], 'confidence': 0.0}
        except:
            return {'present': False, 'bbox': [0, 0, 0, 0], 'confidence': 0.0}
    
    def _get_empty(self) -> Dict[str, Any]:
        return {
            'dealer_name': {'value': None, 'confidence': 0.0},
            'model_name': {'value': None, 'confidence': 0.0},
            'horse_power': {'value': None, 'confidence': 0.0},
            'asset_cost': {'value': None, 'confidence': 0.0},
            'signature': {'present': False, 'bbox': [0, 0, 0, 0], 'confidence': 0.0},
            'stamp': {'present': False, 'bbox': [0, 0, 0, 0], 'confidence': 0.0}
        }


class TeacherEnsemble:
    """
    Ensemble of teacher models for pseudo-label generation.
    Used ONLY during training to create pseudo-labels.
    """
    
    def __init__(self, vlm: VLMExtractor, ocr: OCRExtractor, cv: CVExtractor):
        self.vlm = vlm
        self.ocr = ocr
        self.cv = cv
        self.weights = {'vlm': 0.35, 'ocr': 0.30, 'cv': 0.35}
    
    def generate_pseudo_labels(
        self,
        images: List[np.ndarray],
        language: str
    ) -> tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        """Generate pseudo-labels from all three teachers - RETURNS DICTS NOT NONE"""
        try:
            vlm_results = self.vlm.extract(images, language)
            if vlm_results is None:
                vlm_results = self.vlm._get_empty()
        except Exception as e:
            print(f"    VLM extraction failed: {e}")
            vlm_results = self.vlm._get_empty()
        
        try:
            ocr_results = self.ocr.extract(images, language)
            if ocr_results is None:
                ocr_results = self.ocr._get_empty()
        except Exception as e:
            print(f"    OCR extraction failed: {e}")
            ocr_results = self.ocr._get_empty()
        
        try:
            cv_results = self.cv.extract(images, language)
            if cv_results is None:
                cv_results = self.cv._get_empty()
        except Exception as e:
            print(f"    CV extraction failed: {e}")
            cv_results = self.cv._get_empty()
        
        return vlm_results, ocr_results, cv_results
    
    def merge_predictions(
        self,
        vlm_results: Dict[str, Any],
        ocr_results: Dict[str, Any],
        cv_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Merge predictions using weighted voting"""
        merged = {}
        text_fields = ['dealer_name', 'model_name', 'horse_power', 'asset_cost']
        
        for field in text_fields:
            candidates = []
            for result, source in [(vlm_results, 'vlm'), (ocr_results, 'ocr'), (cv_results, 'cv')]:
                if field in result and result[field]['value'] is not None:
                    weighted_conf = (
                        result[field]['confidence'],
                        self.weights[source]
                    )
                    candidates.append((result[field]['value'], weighted_conf))
            
            if candidates:
                best_value, best_pair = max(
                    candidates,
                    key=lambda x: x[1][0] * x[1][1]
                )
                merged[field] = {
                    'value': best_value,
                    'confidence': best_pair[0]
                }
            else:
                merged[field] = {'value': None, 'confidence': 0.0}
        
        # Visual fields
        for field in ['signature', 'stamp']:
            best_detection = None
            best_score = 0
            for result, source in [(vlm_results, 'vlm'), (ocr_results, 'ocr'), (cv_results, 'cv')]:
                if field in result and result[field]['present']:
                    score = result[field]['confidence'] * self.weights[source]
                    if score > best_score:
                        best_score = score
                        best_detection = result[field]
            
            merged[field] = best_detection if best_detection else {'present': False, 'bbox': [0, 0, 0, 0], 'confidence': 0.0}
        
        # Determine validity
        has_any_text = any(
            merged[f]['value'] is not None
            for f in ['dealer_name', 'model_name', 'horse_power', 'asset_cost']
        )

        has_any_visual = (
            merged['signature']['present'] or
            merged['stamp']['present']
        )

        merged['valid'] = has_any_text or has_any_visual  
        return merged