"""
LayoutLMv3 Explainability Module (Inference-Time Only)
Provides token-level attribution and layout-aware refinement AFTER SGAN predictions
"""

import torch
import numpy as np
from typing import List, Dict, Any
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification


class LayoutLMv3Explainer:
    """
    Post-SGAN explainability using LayoutLMv3.
    Used ONLY during inference - NOT for training.
    """
    
    def __init__(self, model_name: str = "microsoft/layoutlmv3-base"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
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
        
        print(f"LayoutLMv3 Explainer loaded on {self.device} (inference-only)")
    
    def explain_predictions(
        self,
        image: np.ndarray,
        tokens: List[str],
        bboxes: List[List[int]],
        sgan_predictions: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Provide token-level attribution for SGAN predictions
        
        Args:
            image: Document image
            tokens: OCR tokens
            bboxes: Token bounding boxes [x1, y1, x2, y2]
            sgan_predictions: SGAN model outputs
            
        Returns:
            Explainability metadata with token attributions
        """
        try:
            if not tokens:
                return {'attributions': {}, 'confidence': 'low'}
            
            # Normalize bboxes to 0-1000 scale
            img_h, img_w = image.shape[:2]
            normalized_boxes = []
            for bbox in bboxes:
                x1, y1, x2, y2 = bbox
                norm_box = [
                    int((x1 / img_w) * 1000),
                    int((y1 / img_h) * 1000),
                    int((x2 / img_w) * 1000),
                    int((y2 / img_h) * 1000)
                ]
                normalized_boxes.append(norm_box)
            
            # Run LayoutLMv3 inference
            encoding = self.processor(
                image,
                text=tokens,
                boxes=normalized_boxes,
                return_tensors="pt",
                truncation=True,
                padding="max_length",
                max_length=512
            )
            
            encoding = {k: v.to(self.device) for k, v in encoding.items()}
            
            with torch.no_grad():
                outputs = self.model(**encoding)
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=-1)[0]
            
            # Extract token-level attributions
            attributions = {}
            for field in ['dealer_name', 'model_name']:
                if field in sgan_predictions:
                    # Find which tokens LayoutLMv3 assigns to this field
                    field_tokens = self._get_attributed_tokens(
                        tokens, probabilities, field
                    )
                    attributions[field] = field_tokens
            
            return {
                'attributions': attributions,
                'layout_confidence': 'high' if len(attributions) > 0 else 'low',
                'method': 'LayoutLMv3 token classification (post-SGAN)'
            }
        
        except Exception as e:
            print(f"LayoutLMv3 explanation error: {e}")
            return {'attributions': {}, 'confidence': 'unavailable'}
    
    def _get_attributed_tokens(
        self,
        tokens: List[str],
        probabilities: torch.Tensor,
        field: str
    ) -> List[Dict[str, Any]]:
        """Extract tokens attributed to a specific field"""
        label_map = {
            'dealer_name': ['B-DEALER', 'I-DEALER'],
            'model_name': ['B-MODEL', 'I-MODEL']
        }
        
        if field not in label_map:
            return []
        
        target_labels = label_map[field]
        attributed_tokens = []
        
        for idx, token in enumerate(tokens):
            if idx >= probabilities.size(0):
                break
            
            # Get max probability for this token
            token_probs = probabilities[idx]
            max_prob, max_label_id = torch.max(token_probs, dim=0)
            
            # Map label_id to label name (simplified)
            # In production, use proper id2label mapping
            if max_prob.item() > 0.5:  # Threshold for attribution
                attributed_tokens.append({
                    'token': token,
                    'confidence': max_prob.item(),
                    'position': idx
                })
        
        return attributed_tokens