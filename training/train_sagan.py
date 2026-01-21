"""
SGAN Model Trainer with Confidence-Weighted Loss
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
import logging
from typing import Dict

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import LEARNING_RATE, WEIGHT_DECAY


class ConfidenceWeightedLoss(nn.Module):
    """Loss function that weights by pseudo-label confidence"""
    
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
        """Calculate confidence-weighted loss"""
        total_loss = 0.0
        num_fields = 0
        
        # Text fields (simplified: token classification)
        for field in ['dealer_name', 'model_name']:
            if field in labels and field in predictions:
                presence_target = torch.ones_like(predictions[field]['logit'])
                confidence = label_confidences.get(field, 0.5)

                loss = self.bce_loss(
                    predictions[field]['logit'],
                    presence_target
                )

                total_loss += (loss * confidence).mean()
                num_fields += 1

        
        # Numeric fields
        for field in ['horse_power', 'asset_cost']:
            if field not in labels or field not in predictions:
                continue
            if labels[field] is None:
                continue
            if field in labels and field in predictions:
                target_value = labels[field]
                predicted_value = predictions[field]['value']
                
                # Normalize
                if field == 'horse_power':
                    target_value = target_value / 200.0
                else:
                    target_value = target_value / 5000000.0
                
                target_tensor = torch.tensor(
                    [target_value] * predicted_value.size(0),
                    device=predicted_value.device,
                    dtype=torch.float32
                )
                
                confidence = label_confidences.get(field, 0.5)
                loss = self.mse_loss(predicted_value, target_tensor)
                weighted_loss = (loss * confidence).mean()
                
                total_loss += weighted_loss
                num_fields += 1
        
        if num_fields > 0:
            total_loss = total_loss / num_fields
        
        return total_loss


class SGANTrainer:
    """Trainer for SGAN model"""
    
    def __init__(
        self,
        model,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.model = model.to(device)
        self.device = device
        
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY
        )
        
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=100,
            eta_min=1e-6
        )
        
        self.criterion = ConfidenceWeightedLoss()
        
        logging.info(f"Trainer initialized on {device}")
        logging.info(f"Model params: {sum(p.numel() for p in model.parameters()):,}")
    
    def train_epoch(self, dataloader: DataLoader) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch in tqdm(dataloader, desc="Training"):
            token_ids = batch['token_ids'].to(self.device)
            bboxes = batch['bboxes'].to(self.device)
            disagreement = batch['disagreement'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            
            outputs = self.model(token_ids, bboxes, disagreement, attention_mask)
            
            labels = batch['labels']
            confidences = batch['label_confidences']
            
            loss = self.criterion(outputs, labels, confidences)
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches if num_batches > 0 else 0.0
    
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Evaluate on validation set"""
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
        
        return {'loss': total_loss / num_batches if num_batches > 0 else 0.0}
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int,
        save_dir: str
    ):
        """Full training loop"""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            logging.info(f"\nEpoch {epoch + 1}/{num_epochs}")
            
            train_loss = self.train_epoch(train_loader)
            logging.info(f"Train Loss: {train_loss:.4f}")
            
            val_metrics = self.evaluate(val_loader)
            val_loss = val_metrics['loss']
            logging.info(f"Val Loss: {val_loss:.4f}")
            
            self.scheduler.step()
            
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
            
            if (epoch + 1) % 10 == 0:
                checkpoint_path = save_path / f'checkpoint_epoch_{epoch + 1}.pt'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss
                }, checkpoint_path)
        
        logging.info(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")