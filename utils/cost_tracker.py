"""
Cost and Latency Tracking Module
Monitors resource usage and estimates processing costs
"""

from typing import Dict, Any


class CostTracker:
    """Tracks and estimates processing costs"""
    
    def __init__(self):
        """Initialize cost tracker with pricing models"""
        # Cost per operation (in USD)
        self.costs = {
            'vlm_inference': 0.0015,      # Qwen2.5-VL-2B inference
            'ocr_processing': 0.0008,      # PaddleOCR processing
            'cv_detection': 0.0007,        # YOLO inference
            'preprocessing': 0.0002,       # Image preprocessing
            'validation': 0.0001           # Validation overhead
        }
        
        # Cost multipliers for different paths
        self.path_multipliers = {
            'fast': 1.0,       # VLM only
            'standard': 1.6,   # VLM + OCR
            'robust': 2.3      # All extractors
        }
    
    def calculate_cost(
        self,
        extraction_path: str,
        num_pages: int = 1
    ) -> float:
        """
        Calculate total processing cost
        
        Args:
            extraction_path: Extraction strategy used ('fast', 'standard', 'robust')
            num_pages: Number of pages processed
            
        Returns:
            Total cost in USD
        """
        base_cost = self.costs['preprocessing']
        
        if extraction_path == 'fast':
            # VLM only
            processing_cost = self.costs['vlm_inference']
        elif extraction_path == 'standard':
            # VLM + OCR
            processing_cost = (
                self.costs['vlm_inference'] +
                self.costs['ocr_processing']
            )
        else:  # robust
            # All extractors
            processing_cost = (
                self.costs['vlm_inference'] +
                self.costs['ocr_processing'] +
                self.costs['cv_detection']
            )
        
        # Add validation cost
        total_cost = (
            base_cost +
            processing_cost +
            self.costs['validation']
        ) * num_pages
        
        return total_cost
    
    def get_cost_breakdown(
        self,
        extraction_path: str,
        num_pages: int = 1
    ) -> Dict[str, float]:
        """
        Get detailed cost breakdown
        
        Args:
            extraction_path: Extraction strategy used
            num_pages: Number of pages processed
            
        Returns:
            Dictionary with cost breakdown
        """
        breakdown = {
            'preprocessing': self.costs['preprocessing'] * num_pages,
            'validation': self.costs['validation'] * num_pages
        }
        
        if extraction_path in ['fast', 'standard', 'robust']:
            breakdown['vlm_inference'] = self.costs['vlm_inference'] * num_pages
        
        if extraction_path in ['standard', 'robust']:
            breakdown['ocr_processing'] = self.costs['ocr_processing'] * num_pages
        
        if extraction_path == 'robust':
            breakdown['cv_detection'] = self.costs['cv_detection'] * num_pages
        
        breakdown['total'] = sum(breakdown.values())
        
        return breakdown
    
    def estimate_monthly_cost(
        self,
        documents_per_month: int,
        avg_pages_per_doc: int = 1,
        extraction_path: str = 'standard'
    ) -> Dict[str, Any]:
        """
        Estimate monthly processing costs at scale
        
        Args:
            documents_per_month: Expected monthly document volume
            avg_pages_per_doc: Average pages per document
            extraction_path: Extraction strategy to use
            
        Returns:
            Monthly cost estimates
        """
        cost_per_doc = self.calculate_cost(extraction_path, avg_pages_per_doc)
        
        monthly_cost = cost_per_doc * documents_per_month
        yearly_cost = monthly_cost * 12
        
        return {
            'cost_per_document': cost_per_doc,
            'documents_per_month': documents_per_month,
            'monthly_cost': monthly_cost,
            'yearly_cost': yearly_cost,
            'extraction_path': extraction_path
        }