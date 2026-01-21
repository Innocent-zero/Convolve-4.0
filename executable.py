"""
Main Entry Point for Document Extraction System

This demonstrates the complete workflow:
1. Training mode: Train SGAN from pseudo-labels
2. Inference mode: Use trained SGAN with fallback ensemble
3. Validation mode: Validate and normalize results
"""

import logging
from pathlib import Path
import sys

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

from config import CHECKPOINT_DIR
from utils.preprocessing import DocumentPreprocessor
from utils.teacher_extractors import VLMExtractor, OCRExtractor, CVExtractor, TeacherEnsemble
from inference.sgan_extractor import SGANExtractor
from inference.ensemble_inference import InferenceEnsemble
from inference.validator import FieldValidator


def train_mode():
    """Training mode: Train SGAN from scratch using pseudo-labels"""
    logging.info("="*60)
    logging.info("TRAINING MODE")
    logging.info("="*60)
    
    from training.train_pipeline import train_sgan_model, get_image_paths
    
    # Get training data
    data_dir = Path("data")
    image_paths = get_image_paths(data_dir)
    
    if not image_paths:
        logging.error(f"No images found in {data_dir}")
        logging.info("Please add training images to data/ directory")
        return None
    
    logging.info(f"Found {len(image_paths)} training images")
    
    # Initialize components
    preprocessor = DocumentPreprocessor()
    vlm = VLMExtractor()
    ocr = OCRExtractor()
    cv = CVExtractor()
    teacher_ensemble = TeacherEnsemble(vlm, ocr, cv)
    
    # Train
    trained_model = train_sgan_model(
        image_paths,
        teacher_ensemble,
        preprocessor,
        num_iterations=3
    )
    
    logging.info(f"\nModel saved to {CHECKPOINT_DIR}")
    return trained_model


def inference_mode(image_path: str):
    """Inference mode: Use trained SGAN with validation"""
    logging.info("="*60)
    logging.info("INFERENCE MODE")
    logging.info("="*60)
    
    # Check if model exists
    checkpoint_path = CHECKPOINT_DIR / "iteration_3" / "best_model.pt"
    
    if not checkpoint_path.exists():
        logging.error(f"Model not found at {checkpoint_path}")
        logging.info("Please run training first: python main.py --mode train")
        return None
    
    # Load SGAN (student model)
    sgan = SGANExtractor(checkpoint_path=str(checkpoint_path))
    
    # Load fallback extractors (teachers)
    vlm = VLMExtractor()
    ocr = OCRExtractor()
    cv = CVExtractor()
    
    # Create inference ensemble
    ensemble = InferenceEnsemble(
        sgan_extractor=sgan,
        vlm_extractor=vlm,
        ocr_extractor=ocr,
        cv_extractor=cv
    )
    
    # Preprocess document
    preprocessor = DocumentPreprocessor()
    
    logging.info(f"\nProcessing: {image_path}")
    processed = preprocessor.process(image_path)
    
    if not processed['images']:
        logging.error("Failed to process image")
        return None
    
    # Extract fields (adaptive strategy)
    logging.info("Extracting fields...")
    results = ensemble.extract_standard(
        images=processed["images"],
        language=processed["language"]
    )
    
    # Validate and normalize
    validator = FieldValidator()
    validated_results = validator.validate(results, processed['language'])
    
    # Display results
    logging.info("\n" + "="*60)
    logging.info("EXTRACTION RESULTS")
    logging.info("="*60)
    
    logging.info(f"\nMetadata:")
    if '_metadata' in results:
        for key, value in results['_metadata'].items():
            logging.info(f"  {key}: {value}")
    
    logging.info(f"\nExtracted Fields:")
    for field in ['dealer_name', 'model_name', 'horse_power', 'asset_cost']:
        if field in validated_results:
            val = validated_results[field]['value']
            conf = validated_results[field]['confidence']
            logging.info(f"  {field}: {val} (confidence: {conf:.2f})")
    
    logging.info(f"\nVisual Elements:")
    for field in ['signature', 'stamp']:
        if field in validated_results:
            present = validated_results[field]['present']
            conf = validated_results[field]['confidence']
            logging.info(f"  {field}: {'✓' if present else '✗'} (confidence: {conf:.2f})")
    
    logging.info(f"\nOverall Confidence: {validated_results['overall_confidence']:.2f}")
    
    return validated_results


def demo_mode():
    """Demo mode: Show complete workflow"""
    logging.info("="*60)
    logging.info("DEMO MODE - Complete Workflow")
    logging.info("="*60)
    
    # Check for sample data
    sample_image = Path("data/sample_invoice.jpg")
    
    if not sample_image.exists():
        logging.warning(f"Sample image not found at {sample_image}")
        logging.info("\nDemo workflow:")
        logging.info("1. Add training images to data/ directory")
        logging.info("2. Run: python main.py --mode train")
        logging.info("3. Run: python main.py --mode inference --image path/to/invoice.jpg")
        return
    
    # Check if model trained
    checkpoint_path = CHECKPOINT_DIR / "iteration_3" / "best_model.pt"
    
    if not checkpoint_path.exists():
        logging.info("\nStep 1: Training model...")
        train_mode()
    else:
        logging.info("\nModel already trained ✓")
    
    # Run inference
    logging.info("\nStep 2: Running inference...")
    results = inference_mode(str(sample_image))
    
    if results:
        logging.info("\nDemo complete! ✓")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Document Extraction System")
    parser.add_argument(
        '--mode',
        choices=['train', 'inference', 'demo'],
        default='demo',
        help='Operation mode'
    )
    parser.add_argument(
        '--image',
        type=str,
        help='Path to image file (for inference mode)'
    )
    
    args = parser.parse_args()
    
    try:
        if args.mode == 'train':
            train_mode()
        elif args.mode == 'inference':
            if not args.image:
                logging.error("--image required for inference mode")
                sys.exit(1)
            inference_mode(args.image)
        else:  # demo
            demo_mode()
    
    except KeyboardInterrupt:
        logging.info("\nInterrupted by user")
    except Exception as e:
        logging.error(f"Error: {e}", exc_info=True)