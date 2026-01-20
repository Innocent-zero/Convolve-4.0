import torch
import numpy as np
from typing import List, Dict, Any, Optional
from pathlib import Path

from models.spatial_graph_attention import SpatialGraphAttention


class SGANExtractor:
    """
    Primary extractor using our trained Spatial Graph Attention Network.
    
    This is the NOVEL component - a model we trained from scratch using
    pseudo-labels from the VLM/OCR/CV ensemble.
    """
    
    def __init__(self, checkpoint_path: Optional[str] = None):
        """Initialize SGAN extractor"""
        print("    Loading SGAN (Primary Extractor - Trained from Scratch)...")
        
        # Initialize model
        self.model = SpatialGraphAttention(
            vocab_size=10000,
            d_model=256,
            num_heads=8,
            num_layers=6,
            num_fields=4
        )
        
        # Load trained weights if available
        if checkpoint_path and Path(checkpoint_path).exists():
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"      Loaded weights from {checkpoint_path}")
        else:
            print("      Using randomly initialized weights (train first!)")
        
        # Set to eval mode
        self.model.eval()
        
        # Device
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = self.model.to(self.device)
        
        # Vocabulary (simple character-level for demo)
        self.vocab = self._build_vocab()
    
    def _build_vocab(self) -> Dict[str, int]:
        """Build vocabulary"""
        vocab = {'<PAD>': 0, '<UNK>': 1}
        for i in range(256):
            vocab[chr(i)] = i + 2
        return vocab
    
    def _tokenize(self, text: str) -> int:
        """Convert character to token ID"""
        return self.vocab.get(text, self.vocab['<UNK>'])
    
    def _prepare_input(
        self,
        ocr_results: List[Dict[str, Any]]
    ) -> Dict[str, torch.Tensor]:
        """
        Prepare input tensors from OCR results.
        
        Args:
            ocr_results: List of OCR tokens with text, bbox, confidence
            
        Returns:
            Dictionary of input tensors
        """
        max_tokens = 512
        
        # Extract tokens and bboxes
        token_ids = []
        bboxes = []
        disagreement_scores = []
        
        for item in ocr_results[:max_tokens]:
            # Tokenize text
            text = item['text']
            for char in text:
                token_ids.append(self._tokenize(char))
                bboxes.append(item['bbox'])
                # Disagreement: use OCR confidence as proxy
                conf = item.get('confidence', 0.8)
                disagreement_scores.append([conf, conf * 0.9, conf * 0.8])
        
        # Pad to max_tokens
        num_tokens = len(token_ids)
        attention_mask = [True] * num_tokens + [False] * (max_tokens - num_tokens)
        
        token_ids += [0] * (max_tokens - num_tokens)
        bboxes += [[0, 0, 0, 0]] * (max_tokens - num_tokens)
        disagreement_scores += [[0, 0, 0]] * (max_tokens - num_tokens)
        
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
        """
        Decode model outputs to field values.
        
        Args:
            outputs: Raw model outputs
            ocr_results: Original OCR results for text reconstruction
            
        Returns:
            Extracted field values with confidences
        """
        extracted = {}
        
        # Extract text fields (dealer_name, model_name)
        for field in ['dealer_name', 'model_name']:
            if field in outputs:
                # Get attention weights to find relevant tokens
                attn_weights = outputs[field]['attention_weights'][0].cpu().numpy()
                
                # Select top-k tokens by attention
                k = min(10, len(ocr_results))
                top_indices = np.argsort(attn_weights)[-k:][::-1]
                
                # Reconstruct text from selected tokens
                selected_tokens = [ocr_results[i]['text'] for i in top_indices if i < len(ocr_results)]
                value = ' '.join(selected_tokens)
                
                # Get confidence
                confidence = outputs['confidences'][field][0].item()
                
                extracted[field] = {
                    'value': value if value.strip() else None,
                    'confidence': confidence
                }
        
        # Extract numeric fields (horse_power, asset_cost)
        for field in ['horse_power', 'asset_cost']:
            if field in outputs:
                # Denormalize predicted value
                pred_value = outputs[field]['value'][0].item()
                
                if field == 'horse_power':
                    value = int(pred_value * 200)  # Denormalize from [0, 1]
                else:  # asset_cost
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
        language: str,
        ocr_results: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Extract fields using SGAN model.
        
        Args:
            images: Document images
            language: Detected language
            ocr_results: Pre-computed OCR results (optional)
            
        Returns:
            Extracted fields with confidences
        """
        # If no OCR results provided, extract them
        if ocr_results is None:
            # Use PaddleOCR to get tokens and bboxes
            from paddleocr import PaddleOCR
            ocr_engine = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
            
            ocr_results = []
            result = ocr_engine.ocr(images[0], cls=True)
            
            if result and result[0]:
                for line in result[0]:
                    bbox = line[0]
                    # Normalize bbox to [0, 1]
                    h, w = images[0].shape[:2]
                    normalized_bbox = [
                        bbox[0][0] / w, bbox[0][1] / h,
                        bbox[2][0] / w, bbox[2][1] / h
                    ]
                    
                    ocr_results.append({
                        'text': line[1][0],
                        'bbox': normalized_bbox,
                        'confidence': line[1][1]
                    })
        
        # Prepare input
        inputs = self._prepare_input(ocr_results)
        
        # Move to device
        for key in inputs:
            inputs[key] = inputs[key].to(self.device)
        
        # Forward pass
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Decode predictions
        extracted = self._decode_predictions(outputs, ocr_results)
        
        # Add placeholder for visual elements (handled by CV extractor)
        extracted['signature'] = {
            'present': False,
            'bbox': [0, 0, 0, 0],
            'confidence': 0.0
        }
        extracted['stamp'] = {
            'present': False,
            'bbox': [0, 0, 0, 0],
            'confidence': 0.0
        }
        
        return extracted
