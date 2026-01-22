"""
Inference Ensemble with LayoutLMv3 + OCR Rules + YOLO
Adaptive extraction strategy with fallback mechanisms
"""

from typing import List, Dict, Any, Optional
import numpy as np


class InferenceEnsemble:
    """
    Inference ensemble that uses:
    - LayoutLMv3 for text fields (dealer_name, model_name)
    - Rule-based OCR for numeric fields (horse_power, asset_cost)
    - YOLO for visual elements (signature, stamp)
    - SGAN as optional fast path (if available)
    """
    
    def __init__(
        self,
        sgan_extractor=None,
        layoutlm_extractor=None,
        ocr_extractor=None,
        cv_extractor=None
    ):
        """
        Initialize inference ensemble
        
        Args:
            sgan_extractor: Trained SGAN model (optional fast path)
            layoutlm_extractor: LayoutLMv3 extractor
            ocr_extractor: OCR + rules extractor
            cv_extractor: CV/YOLO extractor
        """
        self.sgan = sgan_extractor
        self.layoutlm = layoutlm_extractor
        self.ocr = ocr_extractor
        self.cv = cv_extractor
        
        # Priority weights for confidence-based merging
        self.weights = {
            'sgan': 0.50,
            'layoutlm': 0.40,
            'ocr': 0.30,
            'cv': 0.30
        }
    
    def extract_fast(
        self,
        images: List[np.ndarray],
        language: str
    ) -> Dict[str, Any]:
        """
        Fast path: Try SGAN first, fallback to ensemble
        
        Args:
            images: List of document images
            language: Document language
            
        Returns:
            Extracted fields with metadata
        """
        # Try SGAN if available
        if self.sgan is not None:
            try:
                sgan_results = self.sgan.extract(images, language)
                
                # Check if SGAN results are confident enough
                if self._is_confident(sgan_results, threshold=0.85):
                    sgan_results['_metadata'] = {
                        'extraction_path': 'fast',
                        'extractor': 'sgan',
                        'fallback_used': False
                    }
                    return sgan_results
            except Exception as e:
                print(f"SGAN extraction failed: {e}")
        
        # Fallback to ensemble
        return self.extract_standard(images, language)
    
    def extract_standard(
        self,
        images: List[np.ndarray],
        language: str
    ) -> Dict[str, Any]:
        """
        Standard path: LayoutLMv3 + OCR rules
        
        Args:
            images: List of document images
            language: Document language
            
        Returns:
            Extracted fields with metadata
        """
        results = {}
        
        # First, get OCR tokens for LayoutLMv3
        ocr_tokens = None
        if self.ocr is not None:
            try:
                ocr_tokens = self.ocr.extract_tokens(images)
            except Exception as e:
                print(f"OCR token extraction failed: {e}")
                ocr_tokens = {'tokens': [], 'bboxes': [], 'confidence': []}
        
        # LayoutLMv3 for text fields
        layoutlm_results = {}
        if self.layoutlm is not None and ocr_tokens is not None:
            try:
                layoutlm_results = self.layoutlm._get_empty()    
            except Exception as e:
                print(f"LayoutLMv3 extraction failed: {e}")
                layoutlm_results = {}
        
        # OCR rules for numeric fields
        ocr_results = {}
        if self.ocr is not None:
            try:
                ocr_results = self.ocr.extract(images, language)
            except Exception as e:
                print(f"OCR extraction failed: {e}")
                ocr_results = {}
        
        # CV/YOLO for visual elements
        cv_results = {}
        if self.cv is not None:
            try:
                cv_results = self.cv.extract(images, language)
            except Exception as e:
                print(f"CV extraction failed: {e}")
                cv_results = {}
        
        # Merge results with priority-based logic
        results = self._merge_results(layoutlm_results, ocr_results, cv_results)
        
        results['_metadata'] = {
            'extraction_path': 'standard',
            'extractors_used': ['layoutlm', 'ocr', 'cv'],
            'fallback_used': True
        }
        
        return results
    
    def extract_robust(
        self,
        images: List[np.ndarray],
        language: str
    ) -> Dict[str, Any]:
        """
        Robust path: Use all available extractors with consensus
        
        Args:
            images: List of document images
            language: Document language
            
        Returns:
            Extracted fields with metadata
        """
        all_results = []
        
        # Get OCR tokens first
        ocr_tokens = None
        if self.ocr is not None:
            try:
                ocr_tokens = self.ocr.extract_tokens(images)
            except Exception as e:
                print(f"OCR token extraction failed: {e}")
                ocr_tokens = {'tokens': [], 'bboxes': [], 'confidence': []}
        
        # Try SGAN
        if self.sgan is not None:
            try:
                sgan_results = self.sgan.extract(images, language)
                all_results.append(('sgan', sgan_results))
            except Exception as e:
                print(f"SGAN extraction failed: {e}")
        
        # LayoutLMv3
        if self.layoutlm is not None and ocr_tokens is not None:
            try:
                layoutlm_results = self.layoutlm.extract(images, language, ocr_tokens)
                all_results.append(('layoutlm', layoutlm_results))
            except Exception as e:
                print(f"LayoutLMv3 extraction failed: {e}")
        
        # OCR rules
        if self.ocr is not None:
            try:
                ocr_results = self.ocr.extract(images, language)
                all_results.append(('ocr', ocr_results))
            except Exception as e:
                print(f"OCR extraction failed: {e}")
        
        # CV/YOLO
        if self.cv is not None:
            try:
                cv_results = self.cv.extract(images, language)
                all_results.append(('cv', cv_results))
            except Exception as e:
                print(f"CV extraction failed: {e}")
        
        # Consensus-based merging
        results = self._consensus_merge(all_results)
        
        results['_metadata'] = {
            'extraction_path': 'robust',
            'extractors_used': [name for name, _ in all_results],
            'num_extractors': len(all_results)
        }
        
        return results
    
    def _merge_results(
        self,
        layoutlm_results: Dict[str, Any],
        ocr_results: Dict[str, Any],
        cv_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Merge results with priority-based logic
        
        Priority order:
        1. Rule-based numeric fields (OCR)
        2. LayoutLMv3 text fields
        3. CV/YOLO visual fields
        """
        merged = {}
        
        # Text fields: Prioritize LayoutLMv3
        for field in ['dealer_name', 'model_name']:
            if field in layoutlm_results and layoutlm_results[field]['value'] is not None:
                merged[field] = layoutlm_results[field]
            else:
                merged[field] = {'value': None, 'confidence': 0.0}
        
        # Numeric fields: Use rule-based OCR
        for field in ['horse_power', 'asset_cost']:
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
        
        return merged
    
    def _consensus_merge(
        self,
        all_results: List[tuple[str, Dict[str, Any]]]
    ) -> Dict[str, Any]:
        """
        Merge results using confidence-weighted consensus
        
        Args:
            all_results: List of (extractor_name, results) tuples
            
        Returns:
            Merged results
        """
        merged = {}
        
        # Text fields: Confidence-weighted voting
        for field in ['dealer_name', 'model_name']:
            candidates = []
            for extractor_name, results in all_results:
                if field in results and results[field]['value'] is not None:
                    weighted_conf = results[field]['confidence'] * self.weights.get(extractor_name, 0.3)
                    candidates.append((results[field]['value'], weighted_conf, results[field]['confidence']))
            
            if candidates:
                # Select candidate with highest weighted confidence
                best_value, _, best_conf = max(candidates, key=lambda x: x[1])
                merged[field] = {'value': best_value, 'confidence': best_conf}
            else:
                merged[field] = {'value': None, 'confidence': 0.0}
        
        # Numeric fields: Prioritize rule-based, then consensus
        for field in ['horse_power', 'asset_cost']:
            # First try rule-based OCR
            ocr_value = None
            for extractor_name, results in all_results:
                if extractor_name == 'ocr' and field in results and results[field]['value'] is not None:
                    ocr_value = results[field]
                    break
            
            if ocr_value is not None:
                merged[field] = ocr_value
            else:
                # Fallback to consensus
                candidates = []
                for extractor_name, results in all_results:
                    if field in results and results[field]['value'] is not None:
                        weighted_conf = results[field]['confidence'] * self.weights.get(extractor_name, 0.3)
                        candidates.append((results[field]['value'], weighted_conf, results[field]['confidence']))
                
                if candidates:
                    best_value, _, best_conf = max(candidates, key=lambda x: x[1])
                    merged[field] = {'value': best_value, 'confidence': best_conf}
                else:
                    merged[field] = {'value': None, 'confidence': 0.0}
        
        # Visual fields: Highest confidence
        for field in ['signature', 'stamp']:
            best_detection = None
            best_score = 0
            
            for extractor_name, results in all_results:
                if field in results and results[field]['present']:
                    score = results[field]['confidence'] * self.weights.get(extractor_name, 0.3)
                    if score > best_score:
                        best_score = score
                        best_detection = results[field]
            
            merged[field] = best_detection if best_detection else {'present': False, 'bbox': [0, 0, 0, 0], 'confidence': 0.0}
        
        return merged
    
    def _is_confident(
        self,
        results: Dict[str, Any],
        threshold: float = 0.85
    ) -> bool:
        """
        Check if extraction results meet confidence threshold
        
        Args:
            results: Extraction results
            threshold: Minimum confidence threshold
            
        Returns:
            True if confident, False otherwise
        """
        critical_fields = ['dealer_name', 'model_name', 'asset_cost']
        
        for field in critical_fields:
            if field not in results:
                return False
            if results[field]['value'] is None:
                return False
            if results[field]['confidence'] < threshold:
                return False
        
        return True