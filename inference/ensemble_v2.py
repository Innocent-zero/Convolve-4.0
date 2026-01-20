import torch
import numpy as np
from typing import List, Dict, Any, Optional
from pathlib import Path

from models.spatial_graph_attention import SpatialGraphAttention
from sgan_extractor import SGANExtractor
from utils.extractors import VLMExtractor,OCRExtractor,CVExtractor

class EnsembleExtractorV2:
    """
    Updated ensemble that uses SGAN as primary extractor.
    
    Inference Strategy:
    1. Run SGAN (our trained model)
    2. If SGAN confidence is high (>0.85), use its predictions
    3. Otherwise, fall back to VLM/OCR/CV ensemble
    4. For visual elements (signature/stamp), always use CV
    
    This positions SGAN as the primary learned component while maintaining
    the robustness of the multi-modal ensemble.
    """
    
    def __init__(
        self,
        sgan: SGANExtractor,
        vlm: 'VLMExtractor',
        ocr: 'OCRExtractor',
        cv: 'CVExtractor',
        sgan_confidence_threshold: float = 0.85
    ):
        """
        Initialize ensemble with SGAN as primary.
        
        Args:
            sgan: Our trained SGAN model (PRIMARY)
            vlm: Vision-Language Model (TEACHER/FALLBACK)
            ocr: OCR engine (TEACHER/FALLBACK)
            cv: Computer Vision (for signatures/stamps)
            sgan_confidence_threshold: Min confidence to trust SGAN
        """
        self.sgan = sgan
        self.vlm = vlm
        self.ocr = ocr
        self.cv = cv
        self.sgan_threshold = sgan_confidence_threshold
        
        # Weights for fallback ensemble
        self.fallback_weights = {
            'vlm': 0.50,
            'ocr': 0.35,
            'cv': 0.15
        }
    
    def extract_with_strategy(
        self,
        images: List[np.ndarray],
        language: str,
        strategy: str = 'adaptive'
    ) -> Dict[str, Any]:
        """
        Extract fields using specified strategy.
        
        Strategies:
        - 'sgan_only': Use only SGAN
        - 'ensemble_only': Use only VLM/OCR/CV
        - 'adaptive': Use SGAN if confident, else ensemble (RECOMMENDED)
        
        Args:
            images: Document images
            language: Detected language
            strategy: Extraction strategy
            
        Returns:
            Extracted fields with metadata
        """
        extraction_metadata = {
            'primary_extractor': None,
            'fallback_used': False,
            'sgan_confidence': None
        }
        
        if strategy == 'sgan_only':
            # Use only our trained model
            sgan_results = self.sgan.extract(images, language)
            extraction_metadata['primary_extractor'] = 'sgan'
            
            # Get visual elements from CV
            cv_results = self.cv.extract(images, language)
            sgan_results['signature'] = cv_results['signature']
            sgan_results['stamp'] = cv_results['stamp']
            
            return {
                **sgan_results,
                '_metadata': extraction_metadata
            }
        
        elif strategy == 'ensemble_only':
            # Use traditional ensemble (VLM/OCR/CV)
            vlm_results = self.vlm.extract(images, language)
            ocr_results = self.ocr.extract(images, language)
            cv_results = self.cv.extract(images, language)
            
            merged = self._merge_ensemble_results(
                [vlm_results, ocr_results, cv_results],
                ['vlm', 'ocr', 'cv']
            )
            
            extraction_metadata['primary_extractor'] = 'ensemble'
            merged['_metadata'] = extraction_metadata
            
            return merged
        
        else:  # adaptive (RECOMMENDED)
            # Try SGAN first
            sgan_results = self.sgan.extract(images, language)
            
            # Calculate average confidence
            text_fields = ['dealer_name', 'model_name', 'horse_power', 'asset_cost']
            confidences = [
                sgan_results[f]['confidence']
                for f in text_fields
                if f in sgan_results and sgan_results[f]['value'] is not None
            ]
            
            avg_confidence = np.mean(confidences) if confidences else 0.0
            extraction_metadata['sgan_confidence'] = avg_confidence
            
            # Decision: Trust SGAN or fall back to ensemble?
            if avg_confidence >= self.sgan_threshold:
                # SGAN is confident - use its predictions
                extraction_metadata['primary_extractor'] = 'sgan'
                extraction_metadata['fallback_used'] = False
                
                # Get visual elements from CV
                cv_results = self.cv.extract(images, language)
                sgan_results['signature'] = cv_results['signature']
                sgan_results['stamp'] = cv_results['stamp']
                
                final_results = sgan_results
            
            else:
                # SGAN not confident - fall back to ensemble
                extraction_metadata['primary_extractor'] = 'ensemble'
                extraction_metadata['fallback_used'] = True
                
                vlm_results = self.vlm.extract(images, language)
                ocr_results = self.ocr.extract(images, language)
                cv_results = self.cv.extract(images, language)
                
                # Merge ensemble results
                final_results = self._merge_ensemble_results(
                    [vlm_results, ocr_results, cv_results],
                    ['vlm', 'ocr', 'cv']
                )
                
                # Optionally blend with SGAN predictions (weighted combination)
                for field in text_fields:
                    if field in sgan_results and sgan_results[field]['value'] is not None:
                        # Weighted average: 30% SGAN, 70% ensemble
                        if field in final_results and final_results[field]['value'] is not None:
                            ensemble_conf = final_results[field]['confidence']
                            sgan_conf = sgan_results[field]['confidence']
                            
                            total_weight = ensemble_conf * 0.7 + sgan_conf * 0.3
                            
                            # Use ensemble value but boost confidence if SGAN agrees
                            final_results[field]['confidence'] = min(
                                (ensemble_conf * 0.7 + sgan_conf * 0.3) / total_weight,
                                1.0
                            )
            
            final_results['_metadata'] = extraction_metadata
            return final_results
    
    def _merge_ensemble_results(
        self,
        results: List[Dict[str, Any]],
        sources: List[str]
    ) -> Dict[str, Any]:
        """Merge results from VLM/OCR/CV ensemble (same as before)"""
        merged = {}
        text_fields = ['dealer_name', 'model_name', 'horse_power', 'asset_cost']
        visual_fields = ['signature', 'stamp']
        
        # Merge text fields
        for field in text_fields:
            candidates = []
            for result, source in zip(results, sources):
                if field in result and result[field]['value'] is not None:
                    weighted_conf = result[field]['confidence'] * self.fallback_weights[source]
                    candidates.append((result[field]['value'], weighted_conf))
            
            if candidates:
                best_value, best_conf = max(candidates, key=lambda x: x[1])
                merged[field] = {'value': best_value, 'confidence': best_conf}
            else:
                merged[field] = {'value': None, 'confidence': 0.0}
        
        # Merge visual fields
        for field in visual_fields:
            best_detection = None
            best_score = 0
            
            for result, source in zip(results, sources):
                if field in result and result[field]['present']:
                    score = result[field]['confidence'] * self.fallback_weights[source]
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
    
    # Backwards compatibility methods
    def extract_fast(self, images: List[np.ndarray], language: str) -> Dict[str, Any]:
        """Fast path: SGAN only (our trained model)"""
        return self.extract_with_strategy(images, language, strategy='sgan_only')
    
    def extract_standard(self, images: List[np.ndarray], language: str) -> Dict[str, Any]:
        """Standard path: Adaptive (SGAN + fallback)"""
        return self.extract_with_strategy(images, language, strategy='adaptive')
    
    def extract_robust(self, images: List[np.ndarray], language: str) -> Dict[str, Any]:
        """Robust path: Full ensemble as fallback"""
        return self.extract_with_strategy(images, language, strategy='adaptive')