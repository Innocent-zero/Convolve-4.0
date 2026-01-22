"""
Teacher Extractors (LayoutLMv3, OCR, CV)
These are used ONLY for generating pseudo-labels during training.
They are NOT used during inference with the trained SGAN model.
"""

import re
import numpy as np
from typing import List, Dict, Any, Optional
import cv2
import easyocr
import torch
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification
import warnings
warnings.filterwarnings('ignore')


class LayoutLMv3Extractor:
    """
    LayoutLMv3-based extractor for text fields (dealer_name, model_name)
    Uses token classification with BIO tagging scheme
    """
    
    def __init__(self, model_name: str = "microsoft/layoutlmv3-base"):
        print(f"    Loading LayoutLMv3 ({model_name})...")
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load processor and model
        self.processor = LayoutLMv3Processor.from_pretrained(
            model_name, 
            apply_ocr=False
        )
        self.model = LayoutLMv3ForTokenClassification.from_pretrained(
            model_name,
            num_labels=5  # O, B-DEALER, I-DEALER, B-MODEL, I-MODEL
        )
        self.model.to(self.device)
        self.model.eval()
        
        # Label mapping for BIO scheme
        self.label_map = {
            0: 'O',
            1: 'B-DEALER',
            2: 'I-DEALER',
            3: 'B-MODEL',
            4: 'I-MODEL'
        }
        
        print(f"    LayoutLMv3 loaded on {self.device}")
    
    def extract(self, images: List[np.ndarray], language: str, ocr_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract text fields using LayoutLMv3 token classification
        
        Args:
            images: List of document images
            language: Document language
            ocr_data: Pre-extracted OCR tokens and bboxes
            
        Returns:
            Extracted fields with confidence scores
        """
        try:
            if not images or not ocr_data or not ocr_data.get('tokens'):
                return self._get_empty()
            
            # Use first image
            image = images[0]
            tokens = ocr_data['tokens']
            bboxes = ocr_data['bboxes']
            
            if len(tokens) == 0:
                return self._get_empty()
            
            # Normalize bboxes to LayoutLMv3 format (0-1000 scale)
            img_h, img_w = image.shape[:2]
            normalized_boxes = []
            for bbox in bboxes:
                # bbox is [x1, y1, x2, y2] in pixel coordinates
                x1, y1, x2, y2 = bbox
                # Convert to 0-1000 scale
                norm_box = [
                    int((x1 / img_w) * 1000),
                    int((y1 / img_h) * 1000),
                    int((x2 / img_w) * 1000),
                    int((y2 / img_h) * 1000)
                ]
                normalized_boxes.append(norm_box)
            
            # Prepare inputs for LayoutLMv3
            encoding = self.processor(
                image,
                text=tokens,
                boxes=normalized_boxes,
                return_tensors="pt",
                truncation=True,
                padding="max_length",
                max_length=512
            )
            
            # Move to device
            encoding = {k: v.to(self.device) for k, v in encoding.items()}
            
            # Run inference
            with torch.no_grad():
                outputs = self.model(**encoding)
                logits = outputs.logits
            
            # Get predictions
            predictions = torch.argmax(logits, dim=-1)[0]  # [seq_len]
            probabilities = torch.softmax(logits, dim=-1)[0]  # [seq_len, num_labels]
            
            # Decode predictions to extract fields
            dealer_name, dealer_conf = self._extract_field(
                tokens, predictions, probabilities, 'DEALER'
            )
            
            model_name, model_conf = self._extract_field(
                tokens, predictions, probabilities, 'MODEL'
            )
            
            return {
                'dealer_name': {'value': dealer_name, 'confidence': dealer_conf},
                'model_name': {'value': model_name, 'confidence': model_conf},
                'horse_power': {'value': None, 'confidence': 0.0},
                'asset_cost': {'value': None, 'confidence': 0.0},
                'signature': {'present': False, 'bbox': [0, 0, 0, 0], 'confidence': 0.0},
                'stamp': {'present': False, 'bbox': [0, 0, 0, 0], 'confidence': 0.0}
            }
        
        except Exception as e:
            print(f"    LayoutLMv3 extraction error: {e}")
            return self._get_empty()
    
    def _extract_field(
        self,
        tokens: List[str],
        predictions: torch.Tensor,
        probabilities: torch.Tensor,
        field_type: str
    ) -> tuple[Optional[str], float]:
        """
        Extract field value from BIO-tagged tokens
        
        Args:
            tokens: List of OCR tokens
            predictions: Predicted label IDs
            probabilities: Label probabilities
            field_type: 'DEALER' or 'MODEL'
            
        Returns:
            (field_value, confidence)
        """
        b_label = f'B-{field_type}'
        i_label = f'I-{field_type}'
        
        # Find label IDs
        b_id = None
        i_id = None
        for label_id, label_name in self.label_map.items():
            if label_name == b_label:
                b_id = label_id
            elif label_name == i_label:
                i_id = label_id
        
        if b_id is None:
            return None, 0.0
        
        # Extract tokens with matching labels
        field_tokens = []
        field_probs = []
        
        for idx, pred in enumerate(predictions):
            if idx >= len(tokens):
                break
            
            pred_id = pred.item()
            if pred_id == b_id or pred_id == i_id:
                field_tokens.append(tokens[idx])
                # Get probability for this prediction
                field_probs.append(probabilities[idx, pred_id].item())
        
        if not field_tokens:
            return None, 0.0
        
        # Combine tokens
        field_value = ' '.join(field_tokens)
        
        # Calculate confidence as mean probability
        confidence = float(np.mean(field_probs)) if field_probs else 0.0
        
        return field_value, confidence
    
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
    """OCR-based extractor using EasyOCR + Rule-based extraction"""
    
    def __init__(self):
        print("    Loading Teacher OCR (EasyOCR)...")
        self.reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
        print(f"    EasyOCR loaded (GPU: {torch.cuda.is_available()})")
    
    def extract_tokens(self, images: List[np.ndarray]) -> Dict[str, Any]:
        """
        Extract OCR tokens with bounding boxes (for LayoutLMv3 input)
        
        Returns:
            {
                'tokens': List[str],
                'bboxes': List[List[int]],  # [x1, y1, x2, y2] in pixels
                'confidence': List[float]
            }
        """
        try:
            if not images:
                return {'tokens': [], 'bboxes': [], 'confidence': []}
            
            tokens = []
            bboxes = []
            confidences = []
            
            for image in images:
                if image is None:
                    continue
                
                try:
                    # EasyOCR readtext returns: (bbox, text, confidence)
                    results = self.reader.readtext(image)
                    
                    for (bbox_coords, text, confidence) in results:
                        # bbox_coords is [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                        x_coords = [p[0] for p in bbox_coords]
                        y_coords = [p[1] for p in bbox_coords]
                        
                        # Convert to [x1, y1, x2, y2]
                        bbox = [
                            int(min(x_coords)),
                            int(min(y_coords)),
                            int(max(x_coords)),
                            int(max(y_coords))
                        ]
                        
                        tokens.append(text)
                        bboxes.append(bbox)
                        confidences.append(float(confidence))
                
                except Exception as e:
                    print(f"    OCR error on image: {e}")
                    continue
            
            return {
                'tokens': tokens,
                'bboxes': bboxes,
                'confidence': confidences
            }
        
        except Exception as e:
            print(f"    OCR token extraction error: {e}")
            return {'tokens': [], 'bboxes': [], 'confidence': []}
    
    def extract(self, images: List[np.ndarray], language: str) -> Dict[str, Any]:
        """
        Extract fields using OCR + rule-based methods
        
        Args:
            images: List of document images
            language: Document language
            
        Returns:
            Extracted fields (text + numeric via rules)
        """
        try:
            if not images:
                return self._get_empty()
            
            # Get OCR tokens
            ocr_data = self.extract_tokens(images)
            tokens = ocr_data['tokens']
            
            if not tokens:
                return self._get_empty()
            
            # Combine all text for rule-based extraction
            full_text = ' '.join(tokens)
            
            # Extract text fields using heuristics
            dealer_name = self._extract_dealer_name(tokens, full_text)
            model_name = self._extract_model_name(tokens, full_text)
            
            # Extract numeric fields using rules
            horse_power = self._extract_horse_power(full_text)
            asset_cost = self._extract_asset_cost(full_text)
            
            return {
                'dealer_name': dealer_name,
                'model_name': model_name,
                'horse_power': horse_power,
                'asset_cost': asset_cost,
                'signature': {'present': False, 'bbox': [0, 0, 0, 0], 'confidence': 0.0},
                'stamp': {'present': False, 'bbox': [0, 0, 0, 0], 'confidence': 0.0}
            }
        
        except Exception as e:
            print(f"    OCR extraction error: {e}")
            return self._get_empty()

    def _extract_dealer_name(self, tokens: List[str], text: str) -> Dict[str, Any]:
        """
        Extract dealer name using position + keyword heuristics
        
        Strategy:
        - Look for tokens near top of document (first 20% of tokens)
        - Match against dealer keywords or proper noun patterns
        - Fallback: longest capitalized sequence in top section
        """
        try:
            # Top section heuristic (first 20% of tokens)
            top_section_size = max(5, len(tokens) // 5)
            top_tokens = tokens[:top_section_size]
            
            # Keyword patterns for dealer identification
            dealer_keywords = ['dealer', 'showroom', 'motors', 'auto', 'sales', 'pvt', 'ltd']
            
            # Find tokens matching dealer patterns
            candidates = []
            for i, token in enumerate(top_tokens):
                token_lower = token.lower()
                # Check if contains dealer keywords
                if any(kw in token_lower for kw in dealer_keywords):
                    # Expand context: take 2-3 tokens around match
                    start = max(0, i - 1)
                    end = min(len(top_tokens), i + 3)
                    candidate = ' '.join(top_tokens[start:end])
                    candidates.append((candidate, 0.90))
                # Check if proper noun (capitalized, > 3 chars)
                elif token[0].isupper() and len(token) > 3 and token.isalpha():
                    candidates.append((token, 0.70))
            
            if candidates:
                # Return highest confidence candidate
                best_candidate = max(candidates, key=lambda x: x[1])
                return {'value': best_candidate[0], 'confidence': best_candidate[1]}
            
            # Fallback: longest capitalized sequence in top section
            capitalized_sequences = []
            current_seq = []
            for token in top_tokens:
                if token and token[0].isupper():
                    current_seq.append(token)
                else:
                    if len(current_seq) >= 2:
                        capitalized_sequences.append(' '.join(current_seq))
                    current_seq = []
            
            if capitalized_sequences:
                longest = max(capitalized_sequences, key=len)
                return {'value': longest, 'confidence': 0.65}
            
            return {'value': None, 'confidence': 0.0}
        
        except Exception as e:
            return {'value': None, 'confidence': 0.0}

    def _extract_model_name(self, tokens: List[str], text: str) -> Dict[str, Any]:
        """
        Extract model name using pattern matching
        
        Patterns:
        - Common tractor/vehicle model formats: "ABC-123", "Model X123"
        - Keywords: "model", "variant", "type"
        - Alphanumeric sequences with specific structure
        """
        try:
            # Pattern 1: Model keyword + alphanumeric
            model_pattern_1 = r'(?:model|variant|type)[\s:]*([A-Z0-9\-]{3,15})'
            match = re.search(model_pattern_1, text, re.IGNORECASE)
            if match:
                return {'value': match.group(1), 'confidence': 0.90}
            
            # Pattern 2: Standalone alphanumeric model codes (e.g., "DI-745", "MF375")
            model_pattern_2 = r'\b([A-Z]{2,4}[\-\s]?\d{3,4})\b'
            match = re.search(model_pattern_2, text)
            if match:
                return {'value': match.group(1), 'confidence': 0.85}
            
            # Pattern 3: Search tokens for model-like strings
            for token in tokens:
                # Alphanumeric with dash/hyphen
                if re.match(r'^[A-Z]{2,4}\-\d{2,4}$', token):
                    return {'value': token, 'confidence': 0.80}
                # All caps + numbers
                if re.match(r'^[A-Z]+\d{2,4}$', token) and 4 <= len(token) <= 8:
                    return {'value': token, 'confidence': 0.75}
            
            return {'value': None, 'confidence': 0.0}
        
        except Exception as e:
            return {'value': None, 'confidence': 0.0}
    
    def _extract_horse_power(self, text: str) -> Dict[str, Any]:
        """
        Extract horse power using regex patterns
        
        Patterns:
        - (\d{2,3})\s*(hp|HP|एचपी|हॉर्स)
        - Range: 10-150
        """
        patterns = [
            r'(\d{2,3})\s*(?:hp|HP|एचपी|हॉर्स|horse\s*power)',
            r'(?:power|hp|HP)[\s:]+(\d{2,3})',
            r'(\d{2,3})\s*HP'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    value = int(match.group(1))
                    # Validate range
                    if 10 <= value <= 150:
                        # High confidence if keyword-anchored
                        confidence = 0.95 if 'hp' in match.group(0).lower() else 0.70
                        return {'value': value, 'confidence': confidence}
                except:
                    continue
        
        return {'value': None, 'confidence': 0.0}
    
    def _extract_asset_cost(self, text: str) -> Dict[str, Any]:
        """
        Extract asset cost using regex patterns
        
        Patterns:
        - ₹?\s?\d{1,3}(,\d{3})+
        - Range: 50,000-5,000,000
        """
        patterns = [
            r'(?:cost|price|amount|total|₹)[\s:]*₹?\s?(\d{1,3}(?:,\d{3})+)',
            r'₹\s?(\d{1,3}(?:,\d{3})+)',
            r'(?:Rs\.?|INR)[\s]*(\d{1,3}(?:,\d{3})+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    # Remove commas and convert to int
                    value_str = match.group(1).replace(',', '')
                    value = int(value_str)
                    
                    # Validate range
                    if 50000 <= value <= 5000000:
                        # High confidence if keyword-anchored
                        has_keyword = any(kw in match.group(0).lower() for kw in ['cost', 'price', 'amount', 'total', '₹', 'rs'])
                        confidence = 0.95 if has_keyword else 0.70
                        return {'value': value, 'confidence': confidence}
                except:
                    continue
        
        return {'value': None, 'confidence': 0.0}
    
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
    
    def __init__(self, ocr: OCRExtractor, cv: CVExtractor):
        self.ocr = ocr
        self.cv = cv
        self.weights = {'ocr': 0.50, 'cv': 0.50}

    def generate_pseudo_labels(
        self,
        images: List[np.ndarray],
        language: str
    ) -> tuple[Dict[str, Any], Dict[str, Any]]:
        """Generate pseudo-labels from OCR and CV only"""
        # Get OCR tokens (still needed for SGAN input)
        try:
            ocr_tokens = self.ocr.extract_tokens(images)
        except Exception as e:
            print(f"    OCR token extraction failed: {e}")
            ocr_tokens = {'tokens': [], 'bboxes': [], 'confidence': []}
        
        # OCR rule-based extraction (text + numeric fields)
        try:
            ocr_results = self.ocr.extract(images, language)
            if ocr_results is None:
                ocr_results = self.ocr._get_empty()
        except Exception as e:
            print(f"    OCR extraction failed: {e}")
            ocr_results = self.ocr._get_empty()
        
        # CV extraction (visual elements)
        try:
            cv_results = self.cv.extract(images, language)
            if cv_results is None:
                cv_results = self.cv._get_empty()
        except Exception as e:
            print(f"    CV extraction failed: {e}")
            cv_results = self.cv._get_empty()
        
        return ocr_results, cv_results
    
    def merge_predictions(
        self,
        ocr_results: Dict[str, Any],
        cv_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Merge predictions using OCR + CV only"""
        merged = {}
        
        # All text/numeric fields: Use OCR exclusively
        for field in ['dealer_name', 'model_name', 'horse_power', 'asset_cost']:
            if field in ocr_results and ocr_results[field]['value'] is not None:
                merged[field] = ocr_results[field]
            else:
                merged[field] = {'value': None, 'confidence': 0.0}
        
        # Visual fields: Use CV/YOLO
        for field in ['signature', 'stamp']:
            if field in cv_results and cv_results[field]['present']:
                merged[field] = cv_results[field]
            else:
                merged[field] = {'present': False, 'bbox': [0, 0, 0, 0], 'confidence': 0.0}
        
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