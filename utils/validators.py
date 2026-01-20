"""
Field Validation and Fuzzy Matching Module
Performs validation, normalization, and fuzzy matching against master databases
"""

import re
from typing import Dict, Any, List, Optional
from rapidfuzz import fuzz, process
import numpy as np


class FieldValidator:
    """Validates and normalizes extracted fields"""
    
    def __init__(self):
        """Initialize validator with master databases"""
        # Load master databases (in production, load from files)
        self.dealer_master = self._load_dealer_master()
        self.model_master = self._load_model_master()
        
        # Validation thresholds
        self.fuzzy_threshold = 90  # Minimum fuzzy match score
        self.confidence_threshold = 0.85  # Minimum confidence for acceptance
    
    def _load_dealer_master(self) -> List[str]:
        """Load dealer master database"""
        # In production, load from CSV/database
        return [
            "ABC Tractors Pvt Ltd",
            "XYZ Farm Equipment",
            "Premium Tractor Dealers",
            "Rural Agri Solutions",
            "Mahindra Authorized Dealer",
            "John Deere Sales Center",
            "Swaraj Tractor Hub",
            "Sonalika Farm Equipment",
            "New Holland Agriculture",
            "Escorts Tractor Point"
        ]
    
    def _load_model_master(self) -> List[str]:
        """Load model master database"""
        # In production, load from CSV/database
        return [
            "Mahindra 575 DI",
            "Mahindra 265 DI",
            "Swaraj 744 FE",
            "John Deere 5310",
            "Sonalika DI 750 III",
            "New Holland 3630",
            "Escorts 335",
            "Kubota MU4501"
        ]
    
    def validate(
        self,
        raw_results: Dict[str, Any],
        language: str
    ) -> Dict[str, Any]:
        """
        Validate and normalize all extracted fields
        
        Args:
            raw_results: Raw extraction results
            language: Document language
            
        Returns:
            Validated and normalized results
        """
        validated = {}
        
        # Validate dealer name with fuzzy matching
        validated['dealer_name'] = self._validate_dealer_name(
            raw_results['dealer_name']
        )
        
        # Validate model name with exact matching
        validated['model_name'] = self._validate_model_name(
            raw_results['model_name']
        )
        
        # Validate horse power (numeric)
        validated['horse_power'] = self._validate_numeric(
            raw_results['horse_power'],
            field_name='horse_power',
            min_value=10,
            max_value=200
        )
        
        # Validate asset cost (numeric)
        validated['asset_cost'] = self._validate_numeric(
            raw_results['asset_cost'],
            field_name='asset_cost',
            min_value=100000,
            max_value=5000000
        )
        
        # Validate signature
        validated['signature'] = self._validate_visual_element(
            raw_results['signature'],
            element_type='signature'
        )
        
        # Validate stamp
        validated['stamp'] = self._validate_visual_element(
            raw_results['stamp'],
            element_type='stamp'
        )
        
        # Calculate overall confidence
        validated['overall_confidence'] = self._calculate_overall_confidence(
            validated
        )
        
        return validated
    
    def _validate_dealer_name(self, dealer_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate dealer name with fuzzy matching
        
        Args:
            dealer_data: Extracted dealer name with confidence
            
        Returns:
            Validated dealer name with match information
        """
        if dealer_data['value'] is None:
            return {
                'value': None,
                'confidence': 0.0,
                'matched': False,
                'match_score': 0.0
            }
        
        # Fuzzy match against master database
        best_match, match_score, _ = process.extractOne(
            dealer_data['value'],
            self.dealer_master,
            scorer=fuzz.token_sort_ratio
        )
        
        # Accept if match score >= threshold
        if match_score >= self.fuzzy_threshold:
            return {
                'value': best_match,  # Use standardized name from master
                'original_value': dealer_data['value'],
                'confidence': dealer_data['confidence'],
                'matched': True,
                'match_score': match_score / 100.0
            }
        else:
            return {
                'value': dealer_data['value'],  # Keep original if no good match
                'confidence': dealer_data['confidence'] * 0.8,  # Reduce confidence
                'matched': False,
                'match_score': match_score / 100.0
            }
    
    def _validate_model_name(self, model_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate model name with exact matching
        
        Args:
            model_data: Extracted model name with confidence
            
        Returns:
            Validated model name
        """
        if model_data['value'] is None:
            return {
                'value': None,
                'confidence': 0.0,
                'matched': False
            }
        
        # Try exact match first
        if model_data['value'] in self.model_master:
            return {
                'value': model_data['value'],
                'confidence': model_data['confidence'],
                'matched': True,
                'match_type': 'exact'
            }
        
        # Fuzzy match as fallback
        best_match, match_score, _ = process.extractOne(
            model_data['value'],
            self.model_master,
            scorer=fuzz.ratio
        )
        
        if match_score >= 95:  # Higher threshold for model names
            return {
                'value': best_match,
                'original_value': model_data['value'],
                'confidence': model_data['confidence'],
                'matched': True,
                'match_type': 'fuzzy',
                'match_score': match_score / 100.0
            }
        else:
            return {
                'value': model_data['value'],
                'confidence': model_data['confidence'] * 0.7,
                'matched': False
            }
    
    def _validate_numeric(
        self,
        numeric_data: Dict[str, Any],
        field_name: str,
        min_value: float,
        max_value: float
    ) -> Dict[str, Any]:
        """
        Validate numeric fields
        
        Args:
            numeric_data: Extracted numeric value with confidence
            field_name: Name of the field
            min_value: Minimum acceptable value
            max_value: Maximum acceptable value
            
        Returns:
            Validated numeric value
        """
        if numeric_data['value'] is None:
            return {
                'value': None,
                'confidence': 0.0,
                'valid': False
            }
        
        value = numeric_data['value']
        
        # Type conversion
        try:
            if isinstance(value, str):
                # Remove non-numeric characters
                value = re.sub(r'[^\d.]', '', value)
                value = float(value)
            value = int(value)
        except:
            return {
                'value': None,
                'confidence': 0.0,
                'valid': False,
                'error': 'conversion_failed'
            }
        
        # Range validation
        if min_value <= value <= max_value:
            return {
                'value': value,
                'confidence': numeric_data['confidence'],
                'valid': True
            }
        else:
            return {
                'value': value,
                'confidence': numeric_data['confidence'] * 0.5,
                'valid': False,
                'error': 'out_of_range'
            }
    
    def _validate_visual_element(
        self,
        element_data: Dict[str, Any],
        element_type: str
    ) -> Dict[str, Any]:
        """
        Validate visual elements (signature, stamp)
        
        Args:
            element_data: Detected visual element data
            element_type: Type of element ('signature' or 'stamp')
            
        Returns:
            Validated visual element
        """
        if not element_data['present']:
            return {
                'present': False,
                'bbox': [0, 0, 0, 0],
                'confidence': element_data['confidence'],
                'valid': False
            }
        
        bbox = element_data['bbox']
        
        # Validate bounding box
        if len(bbox) != 4 or any(x < 0 for x in bbox):
            return {
                'present': False,
                'bbox': [0, 0, 0, 0],
                'confidence': 0.0,
                'valid': False,
                'error': 'invalid_bbox'
            }
        
        # Check bbox dimensions
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        
        # Reasonable size checks
        if element_type == 'signature':
            min_size, max_size = 50, 500
        else:  # stamp
            min_size, max_size = 30, 300
        
        if min_size <= width <= max_size and min_size <= height <= max_size:
            return {
                'present': True,
                'bbox': bbox,
                'confidence': element_data['confidence'],
                'valid': True
            }
        else:
            return {
                'present': True,
                'bbox': bbox,
                'confidence': element_data['confidence'] * 0.7,
                'valid': False,
                'error': 'suspicious_size'
            }
    
    def _calculate_overall_confidence(
        self,
        validated_results: Dict[str, Any]
    ) -> float:
        """
        Calculate overall document-level confidence
        
        Args:
            validated_results: All validated fields
            
        Returns:
            Overall confidence score (0-1)
        """
        confidences = []
        weights = {
            'dealer_name': 0.2,
            'model_name': 0.2,
            'horse_power': 0.15,
            'asset_cost': 0.15,
            'signature': 0.15,
            'stamp': 0.15
        }
        
        for field, weight in weights.items():
            if field in validated_results:
                conf = validated_results[field].get('confidence', 0.0)
                confidences.append(conf * weight)
        
        overall = sum(confidences)
        
        # Penalty for missing critical fields
        critical_fields = ['dealer_name', 'model_name', 'asset_cost']
        missing_critical = sum(
            1 for f in critical_fields
            if validated_results.get(f, {}).get('value') is None
        )
        
        if missing_critical > 0:
            overall *= (1 - missing_critical * 0.2)
        
        return float(np.clip(overall, 0.0, 1.0))