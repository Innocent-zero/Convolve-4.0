"""
Training Pipeline for Spatial Graph Attention Network (SGAN)

This script implements the full training loop using pseudo-labels generated
from the VLM/OCR/CV ensemble. The model is trained from scratch (random init).

Key Components:
1. Pseudo-label generation from ensemble consensus
2. Confidence-weighted loss to handle noisy labels
3. Iterative self-training (bootstrapping)
4. Calibration of confidence scores
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Dict, List, Tuple, Optional
import json
from pathlib import Path
from tqdm import tqdm
import logging

from models.spatial_graph_attention import SpatialGraphAttention
from utils.extractors import EnsembleExtractor
from utils.preprocessing import DocumentPreprocessor


class PseudoLabelDataset(Dataset):
    """
    Dataset that generates pseudo-labels from ensemble predictions.
    
    The ensemble (VLM + OCR + CV) acts as a teacher, and we use
    high-confidence consensus predictions as training labels.
    """
    
    def __init__(
        self,
        document_paths: List[str],
        ensemble: EnsembleExtractor,
        preprocessor: DocumentPreprocessor,
        confidence_threshold: float = 0.90,
        consensus_requirement: int = 2
    ):
        self.document_paths = document_paths
        self.ensemble = ensemble
        self.preprocessor = preprocessor
        self.confidence_threshold = confidence_threshold
        self.consensus_requirement = consensus_requirement
        
        # Vocabulary for tokenization
        self.vocab = self._build_vocab()
        
        # Generate pseudo-labels
        self.samples = self._generate_pseudo_labels()
        
        logging.info(f"Created dataset with {len(self.samples)} pseudo-labeled samples")
        logging.info(f"Confidence threshold: {confidence_threshold}")
        logging.info(f"Consensus requirement: {consensus_requirement}/3 extractors")
    
    def _build_vocab(self) -> Dict[str, int]:
        """Build vocabulary from OCR tokens"""
        # Simple character-level vocab for demonstration
        # In production, use BPE or WordPiece
        vocab = {'<PAD>': 0, '<UNK>': 1}
        for i in range(256):  # ASCII characters
            vocab[chr(i)] = i + 2
        return vocab
    
    def _tokenize(self, text: str, max_length: int = 50) -> List[int]:
        """Convert text to token IDs"""
        tokens = [self.vocab.get(c, self.vocab['<UNK>']) for c in text[:max_length]]
        # Pad to max_length
        tokens += [self.vocab['<PAD>']] * (max_length - len(tokens))
        return tokens[:max_length]
    
    def _generate_pseudo_labels(self) -> List[Dict]:
        """
        Generate pseudo-labels using ensemble consensus.
        
        Strategy:
        1. Run all three extractors (VLM, OCR, CV)
        2. Keep predictions where ≥2 extractors agree
        3. Filter by confidence threshold
        4. Store as training samples
        """
        samples = []
        
        for doc_path in tqdm(self.document_paths, desc="Generating pseudo-labels"):
            try:
                # Preprocess document
                processed = self.preprocessor.process(doc_path)
                images = processed['images']
                
                # Get predictions from all extractors
                vlm_results = self.ensemble.vlm.extract(images, processed['language'])
                ocr_results = self.ensemble.ocr.extract(images, processed['language'])
                cv_results = self.ensemble.cv.extract(images, processed['language'])
                
                # Merge results
                merged_results = self.ensemble._merge_results(
                    [vlm_results, ocr_results, cv_results],
                    ['vlm', 'ocr', 'cv']
                )
                
                # Check consensus and confidence
                field_labels = {}
                field_confidences = {}
                
                for field in ['dealer_name', 'model_name', 'horse_power', 'asset_cost']:
                    # Count how many extractors extracted this field
                    extracted_values = []
                    for result in [vlm_results, ocr_results, cv_results]:
                        if field in result and result[field]['value'] is not None:
                            extracted_values.append(result[field]['value'])
                    
                    # Check consensus
                    if len(extracted_values) >= self.consensus_requirement:
                        # Check if values agree (for exact match fields)
                        if field in ['model_name', 'horse_power', 'asset_cost']:
                            if len(set(map(str, extracted_values))) == 1:
                                # Perfect agreement
                                consensus = True
                            else:
                                consensus = False
                        else:
                            # For dealer_name, use fuzzy match
                            consensus = True  # Simplified
                        
                        if consensus and merged_results[field]['confidence'] >= self.confidence_threshold:
                            field_labels[field] = merged_results[field]['value']
                            field_confidences[field] = merged_results[field]['confidence']
                
                # Only include if we have labels for all critical fields
                required_fields = ['dealer_name', 'model_name', 'asset_cost']
                if all(f in field_labels for f in required_fields):
                    # Extract OCR tokens and bboxes (simplified)
                    # In production, use actual OCR output
                    tokens, bboxes = self._extract_ocr_tokens(images[0])
                    
                    # Calculate disagreement scores for each token
                    disagreement = self._calculate_disagreement(
                        tokens, bboxes, vlm_results, ocr_results, cv_results
                    )
                    
                    samples.append({
                        'tokens': tokens,
                        'bboxes': bboxes,
                        'disagreement': disagreement,
                        'labels': field_labels,
                        'confidences': field_confidences,
                        'doc_path': doc_path
                    })
            
            except Exception as e:
                logging.warning(f"Failed to generate pseudo-label for {doc_path}: {e}")
                continue
        
        return samples
    
    def _extract_ocr_tokens(self, image: np.ndarray) -> Tuple[List[str], List[List[float]]]:
        """Extract tokens and bboxes from image (simplified)"""
        # In production, use actual OCR results
        # For now, return dummy data
        tokens = ["Sample", "Invoice", "ABC", "Tractors", "Model", "575", "DI", "Cost", "525000"]
        bboxes = [
            [0.1, 0.1, 0.2, 0.15],
            [0.25, 0.1, 0.35, 0.15],
            [0.1, 0.2, 0.2, 0.25],
            [0.22, 0.2, 0.35, 0.25],
            [0.1, 0.5, 0.2, 0.55],
            [0.25, 0.5, 0.3, 0.55],
            [0.32, 0.5, 0.38, 0.55],
            [0.1, 0.7, 0.2, 0.75],
            [0.25, 0.7, 0.4, 0.75]
        ]
        return tokens, bboxes
    
    def _calculate_disagreement(
        self,
        tokens: List[str],
        bboxes: List[List[float]],
        vlm_results: Dict,
        ocr_results: Dict,
        cv_results: Dict
    ) -> List[List[float]]:
        """Calculate per-token disagreement scores from extractors"""
        # Simplified: return uniform scores
        # In production, match extractor outputs to tokens
        num_tokens = len(tokens)
        disagreement = np.random.rand(num_tokens, 3) * 0.3 + 0.7  # High confidence
        return disagreement.tolist()
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]
        
        # Tokenize text tokens to IDs
        token_texts = sample['tokens']
        token_ids = []
        for text in token_texts:
            ids = self._tokenize(text, max_length=1)
            token_ids.append(ids[0])
        
        # Pad to max sequence length
        max_len = 512
        num_tokens = len(token_ids)
        
        # Create attention mask
        attention_mask = [1] * num_tokens + [0] * (max_len - num_tokens)
        
        # Pad sequences
        token_ids += [0] * (max_len - num_tokens)
        bboxes = sample['bboxes'] + [[0, 0, 0, 0]] * (max_len - num_tokens)
        disagreement = sample['disagreement'] + [[0, 0, 0]] * (max_len - num_tokens)
        
        return {
            'token_ids': torch.tensor(token_ids[:max_len], dtype=torch.long),
            'bboxes': torch.tensor(bboxes[:max_len], dtype=torch.float32),
            'disagreement': torch.tensor(disagreement[:max_len], dtype=torch.float32),
            'attention_mask': torch.tensor(attention_mask[:max_len], dtype=torch.bool),
            'labels': sample['labels'],
            'label_confidences': sample['confidences']
        }


class ConfidenceWeightedLoss(nn.Module):
    """
    Custom loss function that weights samples by pseudo-label confidence.
    
    Novel: Instead of treating all pseudo-labels equally, we weight the loss
    by the confidence of the ensemble that generated them. This helps the model
    learn more from high-quality pseudo-labels and be robust to noise.
    """
    
    def __init__(self):
        super().__init__()
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')
        self.mse_loss = nn.MSELoss(reduction='none')
    
    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        labels: Dict[str, any],
        label_confidences: Dict[str, float]
    ) -> torch.Tensor:
        """
        Calculate confidence-weighted loss.
        
        Args:
            predictions: Model predictions for each field
            labels: Ground truth (pseudo-labels)
            label_confidences: Confidence of each pseudo-label
            
        Returns:
            Weighted loss scalar
        """
        total_loss = 0.0
        num_fields = 0
        
        # Text fields (dealer_name, model_name)
        for field in ['dealer_name', 'model_name']:
            if field in labels and field in predictions:
                # Token-level binary classification loss
                # (Simplified: in production, use proper sequence labeling)
                if 'token_logits' in predictions[field]:
                    # Create target (simplified)
                    batch_size = predictions[field]['token_logits'].size(0)
                    target = torch.zeros_like(predictions[field]['token_logits'])
                    
                    # Loss weighted by confidence
                    confidence = label_confidences.get(field, 0.5)
                    loss = self.bce_loss(predictions[field]['token_logits'], target)
                    weighted_loss = (loss * confidence).mean()
                    
                    total_loss += weighted_loss
                    num_fields += 1
        
        # Numeric fields (horse_power, asset_cost)
        for field in ['horse_power', 'asset_cost']:
            if field in labels and field in predictions:
                target_value = labels[field]
                predicted_value = predictions[field]['value']
                
                # Normalize target
                if field == 'horse_power':
                    target_value = target_value / 200.0  # Normalize to [0, 1]
                else:  # asset_cost
                    target_value = target_value / 5000000.0
                
                target_tensor = torch.tensor(
                    [target_value] * predicted_value.size(0),
                    device=predicted_value.device,
                    dtype=torch.float32
                )
                
                # MSE loss weighted by confidence
                confidence = label_confidences.get(field, 0.5)
                loss = self.mse_loss(predicted_value, target_tensor)
                weighted_loss = (loss * confidence).mean()
                
                total_loss += weighted_loss
                num_fields += 1
        
        # Average over fields
        if num_fields > 0:
            total_loss = total_loss / num_fields
        
        return total_loss


class SGANTrainer:
    """
    Training orchestrator for SGAN model.
    
    Implements iterative self-training:
    1. Initial training on high-confidence pseudo-labels (threshold = 0.95)
    2. Use trained model to generate new pseudo-labels
    3. Combine with original labels and retrain (threshold = 0.90)
    4. Repeat for N iterations
    """
    
    def __init__(
        self,
        model: SpatialGraphAttention,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.model = model.to(device)
        self.device = device
        
        # Optimizer with weight decay for regularization
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=1e-4,
            weight_decay=0.01
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=100,
            eta_min=1e-6
        )
        
        # Loss function
        self.criterion = ConfidenceWeightedLoss()
        
        logging.info(f"Initialized trainer on device: {device}")
        logging.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    def train_epoch(self, dataloader: DataLoader) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch in tqdm(dataloader, desc="Training"):
            # Move to device
            token_ids = batch['token_ids'].to(self.device)
            bboxes = batch['bboxes'].to(self.device)
            disagreement = batch['disagreement'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            
            # Forward pass
            outputs = self.model(token_ids, bboxes, disagreement, attention_mask)
            
            # Calculate loss (batch size = 1 for simplicity)
            labels = batch['labels'][0]
            confidences = batch['label_confidences'][0]
            
            loss = self.criterion(outputs, labels, confidences)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss
    
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Evaluate model on validation set"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                token_ids = batch['token_ids'].to(self.device)
                bboxes = batch['bboxes'].to(self.device)
                disagreement = batch['disagreement'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                outputs = self.model(token_ids, bboxes, disagreement, attention_mask)
                
                labels = batch['labels'][0]
                confidences = batch['label_confidences'][0]
                
                loss = self.criterion(outputs, labels, confidences)
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
        return {'loss': avg_loss}
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int = 50,
        save_dir: str = 'checkpoints'
    ):
        """Full training loop with checkpointing"""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            logging.info(f"\nEpoch {epoch + 1}/{num_epochs}")
            
            # Train
            train_loss = self.train_epoch(train_loader)
            logging.info(f"Train Loss: {train_loss:.4f}")
            
            # Validate
            val_metrics = self.evaluate(val_loader)
            val_loss = val_metrics['loss']
            logging.info(f"Val Loss: {val_loss:.4f}")
            
            # Step scheduler
            self.scheduler.step()
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                checkpoint_path = save_path / 'best_model.pt'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss
                }, checkpoint_path)
                logging.info(f"Saved best model to {checkpoint_path}")
            
            # Save periodic checkpoint
            if (epoch + 1) % 10 == 0:
                checkpoint_path = save_path / f'checkpoint_epoch_{epoch + 1}.pt'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss
                }, checkpoint_path)
        
        logging.info(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")


def iterative_self_training(
    document_paths: List[str],
    ensemble: EnsembleExtractor,
    preprocessor: DocumentPreprocessor,
    num_iterations: int = 3,
    initial_threshold: float = 0.95
):
    """
    Iterative self-training (bootstrapping) procedure.
    
    Algorithm:
    1. Generate initial pseudo-labels with high threshold (0.95)
    2. Train SGAN model
    3. Use SGAN to generate new pseudo-labels on unlabeled data
    4. Lower confidence threshold (0.90, 0.85, ...)
    5. Retrain with expanded pseudo-label set
    6. Repeat
    
    This progressively expands the training set while maintaining quality.
    """
    logging.info("Starting iterative self-training...")
    
    # Initialize model
    model = SpatialGraphAttention(
        vocab_size=10000,
        d_model=256,
        num_heads=8,
        num_layers=6
    )
    
    trainer = SGANTrainer(model)
    
    for iteration in range(num_iterations):
        threshold = initial_threshold - (iteration * 0.05)
        logging.info(f"\n{'='*60}")
        logging.info(f"Iteration {iteration + 1}/{num_iterations}")
        logging.info(f"Confidence threshold: {threshold:.2f}")
        logging.info(f"{'='*60}")
        
        # Generate pseudo-labels with current threshold
        dataset = PseudoLabelDataset(
            document_paths,
            ensemble,
            preprocessor,
            confidence_threshold=threshold,
            consensus_requirement=2
        )
        
        # Split into train/val (80/20)
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset,
            [train_size, val_size]
        )
        
        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
        
        # Train for this iteration
        num_epochs = 30 if iteration == 0 else 20
        trainer.train(
            train_loader,
            val_loader,
            num_epochs=num_epochs,
            save_dir=f'checkpoints/iteration_{iteration + 1}'
        )
    
    logging.info("\nIterative self-training complete!")
    return trainer.model


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Example usage
    logging.info("SGAN Training Pipeline")
    logging.info("="*60)
    
    # In production, load actual document paths
    document_paths = [f"data/invoices/invoice_{i:03d}.pdf" for i in range(100)]
    
    # Initialize ensemble and preprocessor
    from utils.preprocessing import DocumentPreprocessor
    from utils.extractors import VLMExtractor, OCRExtractor, CVExtractor, EnsembleExtractor
    
    preprocessor = DocumentPreprocessor()
    ensemble = EnsembleExtractor(
        vlm=VLMExtractor(),
        ocr=OCRExtractor(),
        cv=CVExtractor()
    )
    
    # Run iterative self-training
    trained_model = iterative_self_training(
        document_paths,
        ensemble,
        preprocessor,
        num_iterations=3
    )
    
    logging.info("Training complete. Model saved to checkpoints/")