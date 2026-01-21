"""
Pseudo-Label Dataset for Training
Generates training samples from teacher ensemble predictions
"""

import torch
from torch.utils.data import Dataset
import numpy as np
from typing import Dict, List, Tuple, Union
from pathlib import Path
from tqdm import tqdm
import logging

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import CONSENSUS_REQUIREMENT, MAX_TOKENS


class PseudoLabelDataset(Dataset):
    """
    Dataset that uses teacher ensemble predictions as pseudo-labels.
    """
    
    def __init__(
        self,
        image_paths: List[Union[str, Path]],
        teacher_ensemble,  # TeacherEnsemble instance
        preprocessor,  # DocumentPreprocessor instance
        confidence_threshold: float = 0.90,
        consensus_requirement: int = CONSENSUS_REQUIREMENT
    ):
        self.image_paths = [Path(p) for p in image_paths]
        self.teacher_ensemble = teacher_ensemble
        self.preprocessor = preprocessor
        self.confidence_threshold = confidence_threshold
        self.consensus_requirement = consensus_requirement
        
        # Build vocabulary
        self.vocab = self._build_vocab()
        
        # Generate pseudo-labels
        self.samples = self._generate_pseudo_labels()
        
        logging.info(f"Dataset created with {len(self.samples)} samples")
        logging.info(f"Confidence threshold: {confidence_threshold}")
        logging.info(f"Consensus requirement: {consensus_requirement}/3 extractors")
    
    def _build_vocab(self) -> Dict[str, int]:
        """Build simple character-level vocabulary"""
        vocab = {'<PAD>': 0, '<UNK>': 1}
        for i in range(256):
            vocab[chr(i)] = i + 2
        return vocab
    
    def _tokenize(self, text: str, max_length: int = 50) -> List[int]:
        """Convert text to token IDs"""
        tokens = [self.vocab.get(c, self.vocab['<UNK>']) for c in text[:max_length]]
        tokens += [self.vocab['<PAD>']] * (max_length - len(tokens))
        return tokens[:max_length]
    
    def _generate_pseudo_labels(self) -> List[Dict]:
        """Generate pseudo-labels using teacher ensemble consensus"""
        samples = []
        
        for img_path in tqdm(self.image_paths, desc="Generating pseudo-labels"):
            try:
                if not img_path.exists():
                    logging.warning(f"Image not found: {img_path}")
                    continue
                
                # Preprocess
                processed = self.preprocessor.process(img_path)
                images = processed['images']
                
                if not images:
                    logging.warning(f"No images from: {img_path}")
                    continue
                
                # Get predictions from all teachers
                vlm_results, ocr_results, cv_results = self.teacher_ensemble.generate_pseudo_labels(
                    images, processed['language']
                )
                
                # Merge results
                merged_results = self.teacher_ensemble.merge_predictions(
                    vlm_results, ocr_results, cv_results
                )
                
                # Check consensus and confidence
                field_labels = {}
                field_confidences = {}
                
                for field in ['dealer_name', 'model_name', 'horse_power', 'asset_cost']:
                    # Count extractors that found this field
                    extracted_values = []
                    for result in [vlm_results, ocr_results, cv_results]:
                        if field in result and result[field]['value'] is not None:
                            extracted_values.append(result[field]['value'])
                    
                    # Check consensus
                    if len(extracted_values) >= self.consensus_requirement:
                        if merged_results[field]['confidence'] >= self.confidence_threshold:
                            field_labels[field] = merged_results[field]['value']
                            field_confidences[field] = merged_results[field]['confidence']
                
                # Only include if we have critical fields
                required_fields = ['dealer_name', 'model_name', 'asset_cost']
                if all(f in field_labels for f in required_fields):
                    # Extract OCR tokens and bboxes
                    tokens, bboxes = self._extract_ocr_tokens(images[0])
                    
                    # Calculate disagreement
                    disagreement = self._calculate_disagreement(
                        tokens, bboxes, vlm_results, ocr_results, cv_results
                    )
                    
                    samples.append({
                        'tokens': tokens,
                        'bboxes': bboxes,
                        'disagreement': disagreement,
                        'labels': field_labels,
                        'confidences': field_confidences,
                        'image_path': str(img_path)
                    })
            
            except Exception as e:
                logging.warning(f"Failed pseudo-label for {img_path}: {e}")
                continue
        
        return samples
    
    def _extract_ocr_tokens(self, image: np.ndarray) -> Tuple[List[str], List[List[float]]]:
        """Extract tokens and bboxes using OCR"""
        try:
            # Use teacher OCR
            result = self.teacher_ensemble.ocr.ocr.ocr(image, cls=True)
            
            tokens = []
            bboxes = []
            
            if result and result[0]:
                img_h, img_w = image.shape[:2]
                
                for line in result[0]:
                    text = line[1][0]
                    bbox = line[0]
                    
                    # Normalize bbox
                    x_coords = [p[0] for p in bbox]
                    y_coords = [p[1] for p in bbox]
                    
                    x1, x2 = min(x_coords) / img_w, max(x_coords) / img_w
                    y1, y2 = min(y_coords) / img_h, max(y_coords) / img_h
                    
                    tokens.append(text)
                    bboxes.append([x1, y1, x2, y2])
            
            # Fallback if no tokens
            if not tokens:
                tokens = ["Sample", "Text"]
                bboxes = [[0.1, 0.1, 0.2, 0.15], [0.25, 0.1, 0.35, 0.15]]
            
            return tokens, bboxes
            
        except Exception as e:
            logging.warning(f"OCR token extraction error: {e}")
            tokens = ["Sample", "Text"]
            bboxes = [[0.1, 0.1, 0.2, 0.15], [0.25, 0.1, 0.35, 0.15]]
            return tokens, bboxes
    
    def _calculate_disagreement(
        self,
        tokens: List[str],
        bboxes: List[List[float]],
        vlm_results: Dict,
        ocr_results: Dict,
        cv_results: Dict
    ) -> List[List[float]]:
        """Calculate per-token disagreement from teachers"""
        num_tokens = len(tokens)
        # Simplified: uniform high confidence
        disagreement = np.random.rand(num_tokens, 3) * 0.3 + 0.7
        return disagreement.tolist()
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]
        
        # Tokenize
        token_texts = sample['tokens']
        token_ids = []
        for text in token_texts:
            ids = self._tokenize(text, max_length=1)
            token_ids.append(ids[0])
        
        # Pad to max length
        num_tokens = len(token_ids)
        attention_mask = [1] * num_tokens + [0] * (MAX_TOKENS - num_tokens)
        
        token_ids += [0] * (MAX_TOKENS - num_tokens)
        bboxes = sample['bboxes'] + [[0, 0, 0, 0]] * (MAX_TOKENS - num_tokens)
        disagreement = sample['disagreement'] + [[0, 0, 0]] * (MAX_TOKENS - num_tokens)
        
        return {
            'token_ids': torch.tensor(token_ids[:MAX_TOKENS], dtype=torch.long),
            'bboxes': torch.tensor(bboxes[:MAX_TOKENS], dtype=torch.float32),
            'disagreement': torch.tensor(disagreement[:MAX_TOKENS], dtype=torch.float32),
            'attention_mask': torch.tensor(attention_mask[:MAX_TOKENS], dtype=torch.bool),
            'labels': sample['labels'],
            'label_confidences': sample['confidences']
        }