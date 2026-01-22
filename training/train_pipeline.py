"""
Main Training Pipeline with Iterative Self-Training
"""

import torch
from torch.utils.data import DataLoader, random_split
from pathlib import Path
from typing import List, Union
import logging

import sys
import os
from config import (
    VOCAB_SIZE, D_MODEL, NUM_HEADS, NUM_LAYERS,
    NUM_EPOCHS_INITIAL, NUM_EPOCHS_ITERATION,
    INITIAL_CONFIDENCE_THRESHOLD, CHECKPOINT_DIR
)
from models.spatial_graph_attention import SpatialGraphAttention
from training.dataset import PseudoLabelDataset
from training.train_sagan import SGANTrainer


def get_image_paths(directory: Union[str, Path]) -> List[Path]:
    """Get all image paths from directory"""
    directory = Path(directory)
    valid_extensions = {'.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp'}
    
    image_paths = []
    for ext in valid_extensions:
        image_paths.extend(directory.glob(f'*{ext}'))
        image_paths.extend(directory.glob(f'*{ext.upper()}'))
    
    return sorted(image_paths)


def train_sgan_model(
    image_paths: List[Union[str, Path]],
    teacher_ensemble,
    preprocessor,
    num_iterations: int = 3,
    initial_threshold: float = INITIAL_CONFIDENCE_THRESHOLD
):
    """
    Iterative self-training procedure.
    
    Args:
        image_paths: List of training image paths
        teacher_ensemble: TeacherEnsemble instance
        preprocessor: DocumentPreprocessor instance
        num_iterations: Number of self-training iterations
        initial_threshold: Initial confidence threshold
    
    Returns:
        Trained model
    """
    logging.info("="*60)
    logging.info("Starting SGAN Training Pipeline")
    logging.info(f"Total images: {len(image_paths)}")
    logging.info(f"Iterations: {num_iterations}")
    logging.info("="*60)
    
    # Initialize model
    model = SpatialGraphAttention(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS
    )
    
    trainer = SGANTrainer(model)
    
    for iteration in range(num_iterations):
        threshold = initial_threshold - (iteration * 0.05)
        threshold = max(threshold, 0.45)
        
        logging.info(f"\n{'='*60}")
        logging.info(f"Iteration {iteration + 1}/{num_iterations}")
        logging.info(f"Confidence threshold: {threshold:.2f}")
        logging.info(f"{'='*60}")
        
        # Limit images for first iteration (demo purposes)
        max_images = 10 if iteration == 0 else len(image_paths)
        iter_image_paths = image_paths[:max_images]
        
        # Generate pseudo-labels
        dataset = PseudoLabelDataset(
            iter_image_paths,
            teacher_ensemble,
            preprocessor,
            confidence_threshold=threshold
        )
        print("DATASET TYPE:", type(dataset))
        print("DATASET CLASS:", dataset.__class__)
        print("DATASET MODULE:", dataset.__class__.__module__)
        print("HAS __len__:", hasattr(dataset, "__len__"))
        print("DIR CONTAINS __len__:", "__len__" in dir(dataset))

        if len(dataset) == 0:
            logging.warning("No samples generated! Check your data.")
            break
        
        # Split train/val
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        
        if val_size == 0:
            val_size = 1
            train_size = len(dataset) - 1
        
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=5,
            shuffle=(len(train_dataset) > 1),
            num_workers=0,          # 🔴 CRITICAL
            pin_memory=False
        )

        
        val_loader = None
        if val_dataset is not None and len(val_dataset) > 0:
            val_loader = DataLoader(
                val_dataset,
                batch_size=5,
                shuffle=False,
                num_workers=0,
                pin_memory=False
            )

        
        # Train
        num_epochs = NUM_EPOCHS_INITIAL if iteration == 0 else NUM_EPOCHS_ITERATION
        save_dir = CHECKPOINT_DIR / f'iteration_{iteration + 1}'
        
        trainer.train(
            train_loader,
            val_loader,
            num_epochs=num_epochs,
            save_dir=str(save_dir)
        )
    
    logging.info("\n" + "="*60)
    logging.info("Training Complete!")
    logging.info("="*60)
    
    return trainer.model


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Import here to avoid circular dependencies
    from utils.preprocessing import DocumentPreprocessor
    from utils.extractors import OCRExtractor, CVExtractor, TeacherEnsemble
    
    # Get image paths
    data_dir = Path("data")
    image_paths = get_image_paths(data_dir)
    
    if not image_paths:
        logging.error(f"No images found in {data_dir}")
        exit(1)
    
    logging.info(f"Found {len(image_paths)} images")
    
    # Initialize teachers and preprocessor (NO LayoutLMv3)
    preprocessor = DocumentPreprocessor()
    ocr = OCRExtractor()
    cv = CVExtractor()
    teacher_ensemble = TeacherEnsemble(ocr, cv)  # Only OCR + CV
    
    # Train
    trained_model = train_sgan_model(
        image_paths,
        teacher_ensemble,
        preprocessor,
        num_iterations=3
    )
    
    logging.info(f"Model saved to {CHECKPOINT_DIR}")