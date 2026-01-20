"""
Intelligent Document AI System - Main Executable
Hybrid Multi-Modal Architecture for Invoice Field Extraction

Author: Hackathon Submission
Date: January 2026
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
import warnings
warnings.filterwarnings('ignore')

# Import utility modules
from utils.preprocessing import DocumentPreprocessor
from utils.extractors import (
    VLMExtractor,
    OCRExtractor,
    CVExtractor,
    EnsembleExtractor
)
from utils.validators import FieldValidator
from utils.cost_tracker import CostTracker

class InvoiceExtractionPipeline:
    """Main orchestrator for invoice field extraction"""
    
    def __init__(self, config_path: str = "config/model_config.yaml"):
        """Initialize the extraction pipeline"""
        print("🚀 Initializing Invoice Extraction System...")
        
        # Initialize components
        self.preprocessor = DocumentPreprocessor()
        self.cost_tracker = CostTracker()
        
        # Initialize extractors
        print("📦 Loading extraction models...")
        self.vlm_extractor = VLMExtractor()
        self.ocr_extractor = OCRExtractor()
        self.cv_extractor = CVExtractor()
        self.ensemble = EnsembleExtractor(
            vlm=self.vlm_extractor,
            ocr=self.ocr_extractor,
            cv=self.cv_extractor
        )
        
        # Initialize validator
        self.validator = FieldValidator()
        
        print("✅ System initialized successfully!\n")
    
    def process_document(self, pdf_path: str) -> Dict[str, Any]:
        """
        Process a single PDF document and extract fields
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Dictionary containing extracted fields and metadata
        """
        start_time = time.time()
        doc_id = Path(pdf_path).stem
        
        print(f"📄 Processing document: {doc_id}")
        
        try:
            # Stage 1: Preprocessing
            print("  ⚙️  Stage 1: Preprocessing...")
            processed_data = self.preprocessor.process(pdf_path)
            images = processed_data['images']
            quality_score = processed_data['quality_score']
            language = processed_data['language']
            
            print(f"      Quality: {quality_score:.2f}, Language: {language}")
            
            # Stage 2: Adaptive Routing & Extraction
            print("  🔍 Stage 2: Multi-path extraction...")
            
            # Determine extraction strategy based on quality
            if quality_score > 0.8:
                # High quality: Use faster path
                extraction_path = "fast"
                raw_results = self.ensemble.extract_fast(
                    images=images,
                    language=language
                )
            elif quality_score > 0.5:
                # Medium quality: Use standard ensemble
                extraction_path = "standard"
                raw_results = self.ensemble.extract_standard(
                    images=images,
                    language=language
                )
            else:
                # Low quality: Use all extractors
                extraction_path = "robust"
                raw_results = self.ensemble.extract_robust(
                    images=images,
                    language=language
                )
            
            print(f"      Path: {extraction_path}")
            
            # Stage 3: Validation & Fuzzy Matching
            print("  ✓  Stage 3: Validation...")
            validated_results = self.validator.validate(
                raw_results=raw_results,
                language=language
            )
            
            # Stage 4: Final assembly
            processing_time = time.time() - start_time
            cost = self.cost_tracker.calculate_cost(
                extraction_path=extraction_path,
                num_pages=len(images)
            )
            
            result = {
                "doc_id": doc_id,
                "fields": {
                    "dealer_name": validated_results['dealer_name']['value'],
                    "model_name": validated_results['model_name']['value'],
                    "horse_power": validated_results['horse_power']['value'],
                    "asset_cost": validated_results['asset_cost']['value'],
                    "signature": validated_results['signature'],
                    "stamp": validated_results['stamp']
                },
                "confidence": validated_results['overall_confidence'],
                "processing_time_sec": round(processing_time, 2),
                "cost_estimate_usd": round(cost, 6),
                "metadata": {
                    "extraction_path": extraction_path,
                    "quality_score": quality_score,
                    "language": language,
                    "num_pages": len(images)
                }
            }
            
            print(f"  ✅ Complete! Confidence: {result['confidence']:.2%}\n")
            
            return result
            
        except Exception as e:
            print(f"  ❌ Error processing {doc_id}: {str(e)}\n")
            return {
                "doc_id": doc_id,
                "error": str(e),
                "processing_time_sec": time.time() - start_time
            }
    
    def process_batch(self, input_dir: str, output_dir: str):
        """
        Process a batch of PDF documents
        
        Args:
            input_dir: Directory containing PDF files
            output_dir: Directory to save results
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Find all PDF files
        pdf_files = list(input_path.glob("*.pdf"))
        
        if not pdf_files:
            print(f"⚠️  No PDF files found in {input_dir}")
            return
        
        print(f"📊 Found {len(pdf_files)} documents to process\n")
        print("=" * 60)
        
        results = []
        successful = 0
        failed = 0
        
        for i, pdf_file in enumerate(pdf_files, 1):
            print(f"\n[{i}/{len(pdf_files)}] Processing: {pdf_file.name}")
            
            result = self.process_document(str(pdf_file))
            results.append(result)
            
            if "error" not in result:
                successful += 1
                # Save individual result
                output_file = output_path / f"{result['doc_id']}.json"
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)
            else:
                failed += 1
        
        # Save batch summary
        summary = {
            "total_documents": len(pdf_files),
            "successful": successful,
            "failed": failed,
            "success_rate": successful / len(pdf_files) if pdf_files else 0,
            "average_processing_time": sum(
                r['processing_time_sec'] for r in results
            ) / len(results),
            "average_cost": sum(
                r.get('cost_estimate_usd', 0) for r in results
            ) / len(results),
            "average_confidence": sum(
                r.get('confidence', 0) for r in results if 'confidence' in r
            ) / successful if successful > 0 else 0,
            "results": results
        }
        
        summary_file = output_path / "batch_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print("\n" + "=" * 60)
        print(f"\n📈 BATCH PROCESSING COMPLETE")
        print(f"   Total Documents: {len(pdf_files)}")
        print(f"   Successful: {successful} ({successful/len(pdf_files)*100:.1f}%)")
        print(f"   Failed: {failed}")
        print(f"   Avg Confidence: {summary['average_confidence']:.2%}")
        print(f"   Avg Processing Time: {summary['average_processing_time']:.2f}s")
        print(f"   Avg Cost: ${summary['average_cost']:.6f}")
        print(f"\n💾 Results saved to: {output_path}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Invoice Field Extraction System'
    )
    parser.add_argument(
        '--input',
        '-i',
        required=True,
        help='Input PDF file or directory'
    )
    parser.add_argument(
        '--output',
        '-o',
        default='output',
        help='Output directory for results (default: output)'
    )
    parser.add_argument(
        '--config',
        '-c',
        default='config/model_config.yaml',
        help='Path to configuration file'
    )
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = InvoiceExtractionPipeline(config_path=args.config)
    
    # Check if input is file or directory
    input_path = Path(args.input)
    
    if input_path.is_file():
        # Process single file
        result = pipeline.process_document(str(input_path))
        
        # Save result
        output_path = Path(args.output)
        output_path.mkdir(parents=True, exist_ok=True)
        output_file = output_path / f"{result['doc_id']}.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Result saved to: {output_file}")
        
    elif input_path.is_dir():
        # Process batch
        pipeline.process_batch(str(input_path), args.output)
    else:
        print(f"❌ Error: {args.input} is not a valid file or directory")
        sys.exit(1)


if __name__ == "__main__":
    main()