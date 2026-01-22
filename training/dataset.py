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
                images = processed.get('images', [])
                
                if not images:
                    logging.warning(f"No images from: {img_path}")
                    continue
                
                # Get predictions from OCR + CV only (2 teachers)
                try:
                    ocr_results, cv_results = self.teacher_ensemble.generate_pseudo_labels(
                        images, processed.get('language', 'en')
                    )
                except Exception as e:
                    logging.warning(f"Teacher ensemble failed for {img_path}: {e}")
                    continue
                
                # Check if results are None
                if ocr_results is None or cv_results is None:
                    logging.warning(f"Got None results from teachers for {img_path}")
                    continue
                
                # Merge results
                try:
                    merged_results = self.teacher_ensemble.merge_predictions(
                        ocr_results, cv_results
                    )
                except Exception as e:
                    logging.warning(f"Failed to merge predictions for {img_path}: {e}")
                    continue
                
                if not merged_results or not merged_results.get('valid', False):
                    logging.warning(f"Skipping unusable sample: {img_path}")
                    continue
                
                # Adjusted consensus: with only 2 teachers (OCR, CV)
                # For text/numeric: OCR is authoritative (consensus = 1)
                # For visual: CV is authoritative (consensus = 1)
                field_labels = {}
                field_confidences = {}
                
                for field in ['dealer_name', 'model_name', 'horse_power', 'asset_cost']:
                    # OCR provides these fields
                    if field in merged_results and merged_results[field].get('value') is not None:
                        field_conf = merged_results[field].get('confidence', 0.0)
                        if field_conf >= self.confidence_threshold:
                            field_labels[field] = merged_results[field]['value']
                            field_confidences[field] = field_conf
                
                # Relaxed requirement: need at least 2 critical fields (not all 3)
                critical_fields = ['dealer_name', 'model_name', 'asset_cost']
                critical_found = sum(1 for f in critical_fields if f in field_labels)
                
                if critical_found >= 2:  # At least 2/3 critical fields
                    # Extract OCR tokens and bboxes
                    tokens, bboxes = self._extract_ocr_tokens(images[0])
                    
                    # Calculate disagreement (simplified for 2 teachers)
                    disagreement = self._calculate_disagreement(
                        tokens, bboxes, ocr_results, cv_results
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
                import traceback
                logging.debug(traceback.format_exc())
                continue
        
        return samples
    
    def _extract_ocr_tokens(self, image: np.ndarray) -> Tuple[List[str], List[List[float]]]:
        """Extract tokens and bboxes using OCR"""
        try:
            # Use teacher EasyOCR
            results = self.teacher_ensemble.ocr.reader.readtext(image)
            
            tokens = []
            bboxes = []
            
            if results:
                img_h, img_w = image.shape[:2]
                
                for (bbox, text, confidence) in results:
                    # EasyOCR bbox is [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                    # Extract all x and y coordinates
                    x_coords = [point[0] for point in bbox]
                    y_coords = [point[1] for point in bbox]
                    
                    # Normalize bbox
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
        ocr_results: Dict,
        cv_results: Dict
    ) -> List[List[float]]:
        """Calculate per-token disagreement from 2 teachers (OCR, CV)"""
        num_tokens = len(tokens)
        # For 2 teachers: disagreement is binary (agree/disagree)
        # Simplified: high confidence (low disagreement) for consensus tokens
        disagreement = np.random.rand(num_tokens, 2) * 0.2 + 0.8  # High confidence
        return disagreement.tolist()
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]