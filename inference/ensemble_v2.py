"""
Inference Ensemble
Combines SGAN (student) with teacher extractors as fallback
"""

import numpy as np
from typing import List, Dict, Any

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import SGAN_CONFIDENCE_THRESHOLD


class InferenceEnsemble:
    """
    Inference-time ensemble using SGAN as primary extractor.
    
    Strategy:
    1. Run SGAN (trained student model)
    2. If SGAN confidence is high, use its predictions
    3. Otherwise, fall back to teacher ensemble (VLM/OCR/CV)
    4. For visual elements (signature/stamp), always use CV
    """
    
    def __init__(
        self,
        sgan_extractor,  # SGANExtractor
        vlm_extractor,   # VLMExtractor (teacher)
        ocr_extractor,   # OCRExtractor (teacher)
        cv_extractor,    # CVExtractor (teacher)
        sgan_confidence_threshold: float = SGAN_CONFIDENCE_THRESHOLD
    ):
        self.sgan = sgan_extractor
        self.vlm = vlm_extractor
        self.ocr = ocr_extractor
        self.cv = cv_extractor
        self.sgan_threshold = sgan_confidence_threshold
        
        # Fallback ensemble weights
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
        - 'sgan_only': Use only SGAN (student)
        - 'teachers_only': Use only teacher ensemble
        - 'adaptive': Use SGAN if confident, else teachers (RECOMMENDED)
        
        Args:
            images: Document images
            language: Detected language
            strategy: Extraction strategy
            
        Returns:
            Extracted fields with metadata
        """
        metadata = {
            'primary_extractor': None,
            'fallback_used': False,
            'sgan_confidence': None
        }
        
        if strategy == 'sgan_only':
            # Student only
            sgan_results = self.sgan.extract(images, language)
            metadata['primary_extractor'] = 'sgan'
            
            # Get visual elements from CV
            cv_results = self.cv.extract(images, language)
            sgan_results['signature'] = cv_results['signature']
            sgan_results['stamp'] = cv_results['stamp']
            
            return {**sgan_results, '_metadata': metadata}
        
        elif strategy == 'teachers_only':
            # Teachers only (for comparison)
            vlm_results = self.vlm.extract(images, language)
            ocr_results = self.ocr.extract(images, language)
            cv_results = self.cv.extract(images, language)
            
            merged = self._merge_teacher_results(vlm_results, ocr_results, cv_results)
            metadata['primary_extractor'] = 'teachers'
            merged['_metadata'] = metadata
            
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
            metadata['sgan_confidence'] = float(avg_confidence)
            
            # Decision: trust SGAN or fall back?
            if avg_confidence >= self.sgan_threshold:
                # SGAN is confident
                metadata['primary_extractor'] = 'sgan'
                metadata['fallback_used'] = False
                
                # Get visual elements
                cv_results = self.cv.extract(images, language)
                sgan_results['signature'] = cv_results['signature']
                sgan_results['stamp'] = cv_results['stamp']
                
                final_results = sgan_results
            
            else:
                # Fall back to teachers
                metadata['primary_extractor'] = 'teachers'
                metadata['fallback_used'] = True
                
                vlm_results = self.vlm.extract(images, language)
                ocr_results = self.ocr.extract(images, language)
                cv_results = self.cv.extract(images, language)
                
                final_results = self._merge_teacher_results(vlm_results, ocr_results, cv_results)
                
                # Optionally blend with SGAN (weighted)
                for field in text_fields:
                    if field in sgan_results and sgan_results[field]['value'] is not None:
                        if field in final_results and final_results[field]['value'] is not None:
                            # Boost confidence if SGAN agrees
                            ensemble_conf = final_results[field]['confidence']
                            sgan_conf = sgan_results[field]['confidence']
                            
                            final_results[field]['confidence'] = min(
                                (ensemble_conf * 0.7 + sgan_conf * 0.3),
                                1.0
                            )
            
            final_results['_metadata'] = metadata
            return final_results
    
    def _merge_teacher_results(
        self,
        vlm_results: Dict[str, Any],
        ocr_results: Dict[str, Any],
        cv_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Merge teacher ensemble results"""
        merged = {}
        text_fields = ['dealer_name', 'model_name', 'horse_power', 'asset_cost']
        
        # Text fields: weighted voting
        for field in text_fields:
            candidates = []
            for result, source in [(vlm_results, 'vlm'), (ocr_results, 'ocr'), (cv_results, 'cv')]:
                if field in result and result[field]['value'] is not None:
                    weighted_conf = result[field]['confidence'] * self.fallback_weights[source]
                    candidates.append((result[field]['value'], weighted_conf))
            
            if candidates:
                best_value, best_conf = max(candidates, key=lambda x: x[1])
                merged[field] = {'value': best_value, 'confidence': best_conf}
            else:
                merged[field] = {'value': None, 'confidence': 0.0}
        
        # Visual fields: best detection
        for field in ['signature', 'stamp']:
            best_detection = None
            best_score = 0
            
            for result, source in [(vlm_results, 'vlm'), (ocr_results, 'ocr'), (cv_results, 'cv')]:
                if field in result and result[field]['present']:
                    score = result[field]['confidence'] * self.fallback_weights[source]
                    if score > best_score:
                        best_score = score
                        best_detection = result[field]
            
            if best_detection:
                merged[field] = best_detection
            else:
                merged[field] = {'present': False, 'bbox': [0, 0, 0, 0], 'confidence': 0.0}
        
        return merged
    
    # Convenience methods (backwards compatibility)
    def extract_fast(self, images: List[np.ndarray], language: str) -> Dict[str, Any]:
        """Fast path: SGAN only"""
        return self.extract_with_strategy(images, language, strategy='sgan_only')
    
    def extract_standard(self, images: List[np.ndarray], language: str) -> Dict[str, Any]:
        """Standard path: Adaptive (SGAN + fallback)"""
        return self.extract_with_strategy(images, language, strategy='adaptive')
    
    def extract_robust(self, images: List[np.ndarray], language: str) -> Dict[str, Any]:
        """Robust path: Same as adaptive"""
        return self.extract_with_strategy(images, language, strategy='adaptive')