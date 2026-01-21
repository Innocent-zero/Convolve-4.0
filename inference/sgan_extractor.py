"""
SGAN Extractor for Inference
Uses the trained student model for field extraction
"""

import torch
import numpy as np
from typing import List, Dict, Any, Optional
from pathlib import Path
from paddleocr import PaddleOCR

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.spatial_graph_attention import SpatialGraphAttention
from config import VOCAB_SIZE, D_MODEL, NUM_HEADS, NUM_LAYERS, MAX_TOKENS


class SGANExtractor:
    """
    Primary extractor using trained SGAN model.
    This is the STUDENT model trained from teacher pseudo-labels.
    """
    
    def __init__(self, checkpoint_path: Optional[str] = None):
        print("    Loading SGAN (Student Model)...")
        
        # Initialize model
        self.model = SpatialGraphAttention(
            vocab_size=VOCAB_SIZE,
            d_model=D_MODEL,
            num_heads=NUM_HEADS,
            num_layers=NUM_LAYERS,
            num_fields=4
        )
        
        # Load trained weights
        if checkpoint_path and Path(checkpoint_path).exists():
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"      Loaded weights from {checkpoint_path}")
        else:
            print("      WARNING: Using random weights! Train the model first.")
        
        self.model.eval()
        
        # Device
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = self.model.to(self.device)
        
        # Vocabulary
        self.vocab = self._build_vocab()
        
        # OCR for token extraction
        self.ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
    
    def _build_vocab(self) -> Dict[str, int]:
        """Build vocabulary"""
        vocab = {'<PAD>': 0, '<UNK>': 1}
        for i in range(256):
            vocab[chr(i)] = i + 2
        return vocab
    
    def _tokenize(self, text: str) -> int:
        """Convert character to token ID"""
        return self.vocab.get(text, self.vocab['<UNK>'])
    
    def _extract_ocr_tokens(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Extract OCR tokens with bboxes"""
        try:
            result = self.ocr.ocr(image, cls=True)
            
            ocr_results = []
            if result and result[0]:
                img_h, img_w = image.shape[:2]
                
                for line in result[0]:
                    bbox = line[0]
                    x_coords = [p[0] for p in bbox]
                    y_coords = [p[1] for p in bbox]
                    
                    normalized_bbox = [
                        min(x_coords) / img_w,
                        min(y_coords) / img_h,
                        max(x_coords) / img_w,
                        max(y_coords) / img_h
                    ]
                    
                    ocr_results.append({
                        'text': line[1][0],
                        'bbox': normalized_bbox,
                        'confidence': line[1][1]
                    })
            
            return ocr_results
        except Exception as e:
            print(f"OCR error: {e}")
            return []
    
    def _prepare_input(self, ocr_results: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Prepare input tensors"""
        token_ids = []
        bboxes = []
        disagreement_scores = []
        
        for item in ocr_results[:MAX_TOKENS]:
            text = item['text']
            for char in text:
                token_ids.append(self._tokenize(char))
                bboxes.append(item['bbox'])
                conf = item.get('confidence', 0.8)
                disagreement_scores.append([conf, conf * 0.9, conf * 0.8])
        
        # Pad
        num_tokens = len(token_ids)
        attention_mask = [True] * num_tokens + [False] * (MAX_TOKENS - num_tokens)
        
        token_ids += [0] * (MAX_TOKENS - num_tokens)
        bboxes += [[0, 0, 0, 0]] * (MAX_TOKENS - num_tokens)
        disagreement_scores += [[0, 0, 0]] * (MAX_TOKENS - num_tokens)
        
        return {
            'token_ids': torch.tensor([token_ids], dtype=torch.long),
            'bboxes': torch.tensor([bboxes], dtype=torch.float32),
            'disagreement': torch.tensor([disagreement_scores], dtype=torch.float32),
            'attention_mask': torch.tensor([attention_mask], dtype=torch.bool)
        }
    
    def _decode_predictions(
        self,
        outputs: Dict[str, Any],
        ocr_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Decode model outputs to field values"""
        extracted = {}
        
        # Text fields
        for field in ['dealer_name', 'model_name']:
            if field in outputs:
                attn_weights = outputs[field]['attention_weights'][0].cpu().numpy()
                k = min(10, len(ocr_results))
                top_indices = np.argsort(attn_weights)[-k:][::-1]
                
                selected_tokens = [ocr_results[i]['text'] for i in top_indices if i < len(ocr_results)]
                value = ' '.join(selected_tokens)
                
                confidence = outputs['confidences'][field][0].item()
                
                extracted[field] = {
                    'value': value if value.strip() else None,
                    'confidence': confidence
                }
        
        # Numeric fields
        for field in ['horse_power', 'asset_cost']:
            if field in outputs:
                pred_value = outputs[field]['value'][0].item()
                
                if field == 'horse_power':
                    value = int(pred_value * 200)
                else:
                    value = int(pred_value * 5000000)
                
                confidence = outputs['confidences'][field][0].item()
                
                extracted[field] = {
                    'value': value if value > 0 else None,
                    'confidence': confidence
                }
        
        return extracted
    
    def extract(
        self,
        images: List[np.ndarray],
        language: str
    ) -> Dict[str, Any]:
        """
        Extract fields using trained SGAN model.
        
        Args:
            images: Document images
            language: Detected language
            
        Returns:
            Extracted fields with confidences
        """
        # Extract OCR tokens
        ocr_results = self._extract_ocr_tokens(images[0])
        
        if not ocr_results:
            # Return empty if no OCR
            return {
                'dealer_name': {'value': None, 'confidence': 0.0},
                'model_name': {'value': None, 'confidence': 0.0},
                'horse_power': {'value': None, 'confidence': 0.0},
                'asset_cost': {'value': None, 'confidence': 0.0},
                'signature': {'present': False, 'bbox': [0, 0, 0, 0], 'confidence': 0.0},
                'stamp': {'present': False, 'bbox': [0, 0, 0, 0], 'confidence': 0.0}
            }
        
        # Prepare input
        inputs = self._prepare_input(ocr_results)
        
        # Move to device
        for key in inputs:
            inputs[key] = inputs[key].to(self.device)
        
        # Forward pass
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Decode
        extracted = self._decode_predictions(outputs, ocr_results)
        
        # Placeholder for visual elements (handled by CV in ensemble)
        extracted['signature'] = {'present': False, 'bbox': [0, 0, 0, 0], 'confidence': 0.0}
        extracted['stamp'] = {'present': False, 'bbox': [0, 0, 0, 0], 'confidence': 0.0}
        
        return extracted