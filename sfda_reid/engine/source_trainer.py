import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing import Any
import logging

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    class SummaryWriter:  # type: ignore[override]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def add_scalar(self, *args: Any, **kwargs: Any) -> None:
            pass

        def close(self) -> None:
            pass

class SourceTrainer:
    """
    Standard supervised training on source domain.
    Uses SupConLoss + CrossEntropyLoss with label smoothing.
    Cosine LR schedule with warmup (10 epochs).
    Save best checkpoint by Rank-1 on source validation split.
    Log to TensorBoard: loss, lr, Rank-1.
    """
    def __init__(self, model: nn.Module, train_loader, val_loader, cfg: Any):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.writer = SummaryWriter(log_dir=f"outputs/runs/source_{cfg.source.dataset}")
        self.logger = logging.getLogger("SourceTrainer")
        self.criterion_ce = nn.CrossEntropyLoss(label_smoothing=0.1)
        from ..losses.contrastive import SupConLoss
        self.criterion_supcon = SupConLoss(temperature=cfg.memory_bank.temperature)
        self.optimizer = optim.Adam(self.model.parameters(), lr=cfg.source.lr)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=cfg.source.num_epochs)
        self.best_rank1 = 0.0
        
        # Log training setup
        self.logger.info("=" * 80)
        self.logger.info("SOURCE TRAINER INITIALIZED")
        self.logger.info("=" * 80)
        self.logger.info(f"Model: {cfg.source.backbone}")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Dataset: {cfg.source.dataset}")
        self.logger.info(f"Number of classes: {cfg.source.num_classes}")
        self.logger.info(f"Batch size: {cfg.source.batch_size}")
        self.logger.info(f"Number of epochs: {cfg.source.num_epochs}")
        self.logger.info(f"Learning rate: {cfg.source.lr}")
        self.logger.info(f"Training samples: {len(train_loader) * cfg.source.batch_size}")
        self.logger.info(f"Validation samples: {len(val_loader) * 256}")
        self.logger.info(f"Batches per epoch: {len(train_loader)}")
        self.logger.info("=" * 80)

    def train(self):
        self.logger.info(f"Starting training for {self.cfg.source.num_epochs} epochs...")
        self.logger.info("=" * 80)
        
        for epoch in range(self.cfg.source.num_epochs):
            self.model.train()
            total_loss, total_ce, total_supcon = 0, 0, 0
            batch_count = 0
            
            self.logger.info(f"[Epoch {epoch+1}/{self.cfg.source.num_epochs}] Starting training...")
            
            for batch_idx, batch in enumerate(self.train_loader):
                images = batch['image'].to(self.device)
                labels = batch['pid'].to(self.device)
                feats, logits = self.model(images)
                loss_ce = self.criterion_ce(logits, labels)
                loss_supcon = self.criterion_supcon(feats, labels)
                loss = loss_ce + loss_supcon
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                total_ce += loss_ce.item()
                total_supcon += loss_supcon.item()
                batch_count += 1
                
                # Log batch-level progress
                if (batch_idx + 1) % 50 == 0:
                    current_lr = self.optimizer.param_groups[0]['lr']
                    avg_batch_loss = total_loss / batch_count
                    avg_batch_ce = total_ce / batch_count
                    avg_batch_supcon = total_supcon / batch_count
                    self.logger.info(
                        f"  [Batch {batch_idx+1}/{len(self.train_loader)}] "
                        f"Loss: {avg_batch_loss:.4f} (CE: {avg_batch_ce:.4f}, SupCon: {avg_batch_supcon:.4f}) "
                        f"LR: {current_lr:.6f}"
                    )
            
            self.scheduler.step()
            avg_loss = total_loss / len(self.train_loader)
            avg_ce = total_ce / len(self.train_loader)
            avg_supcon = total_supcon / len(self.train_loader)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            self.writer.add_scalar('Loss/total', avg_loss, epoch)
            self.writer.add_scalar('Loss/ce', avg_ce, epoch)
            self.writer.add_scalar('Loss/supcon', avg_supcon, epoch)
            self.writer.add_scalar('LR', current_lr, epoch)
            
            # Log epoch summary
            self.logger.info(
                f"[Epoch {epoch+1}/{self.cfg.source.num_epochs}] "
                f"Train Loss: {avg_loss:.4f} (CE: {avg_ce:.4f}, SupCon: {avg_supcon:.4f}) "
                f"LR: {current_lr:.6f}"
            )
            
            # Evaluation
            if (epoch + 1) % self.cfg.eval_every == 0:
                self.logger.info(f"[Epoch {epoch+1}] Running evaluation on validation set...")
                rank1 = self.evaluate(epoch)
                self.logger.info(
                    f"[Epoch {epoch+1}] Evaluation - Rank-1: {rank1:.4f} "
                    f"(Best: {self.best_rank1:.4f})"
                )
                if rank1 > self.best_rank1:
                    self.best_rank1 = rank1
                    self.save_checkpoint(epoch, best=True)
                    self.logger.info(f"[Epoch {epoch+1}] ✓ New best model saved! Rank-1: {rank1:.4f}")
        
        self.logger.info("=" * 80)
        self.logger.info("Training completed!")
        self.logger.info(f"Best Rank-1: {self.best_rank1:.4f}")
        self.logger.info("=" * 80)
        self.writer.close()

    def evaluate(self, epoch: int) -> float:
        self.model.eval()
        from ..engine.evaluator import ReIDEvaluator
        evaluator = ReIDEvaluator()
        
        self.logger.info(f"  Extracting features from gallery set...")
        self.logger.info(f"  Extracting features from query set...")
        
        metrics = evaluator.evaluate(self.model, self.val_loader, self.val_loader)
        
        rank1 = metrics['rank1']
        rank5 = metrics.get('rank5', 0)
        rank10 = metrics.get('rank10', 0)
        mAP = metrics.get('mAP', 0)
        
        self.logger.info(f"  Evaluation metrics:")
        self.logger.info(f"    - Rank-1: {rank1:.4f}")
        self.logger.info(f"    - Rank-5: {rank5:.4f}")
        self.logger.info(f"    - Rank-10: {rank10:.4f}")
        self.logger.info(f"    - mAP: {mAP:.4f}")
        
        self.writer.add_scalar('Metrics/rank1', rank1, epoch)
        self.writer.add_scalar('Metrics/rank5', rank5, epoch)
        self.writer.add_scalar('Metrics/rank10', rank10, epoch)
        self.writer.add_scalar('Metrics/mAP', mAP, epoch)
        
        return rank1

    def save_checkpoint(self, epoch: int, best: bool = False):
        state = {
            'epoch': epoch,
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'best_rank1': self.best_rank1,
            'cfg': self.cfg,
        }
        out_dir = self.cfg.source.output_dir
        fname = f"best.pth" if best else f"epoch_{epoch+1}.pth"
        checkpoint_path = f"{out_dir}/{fname}"
        torch.save(state, checkpoint_path)
        
        if best:
            self.logger.info(f"  Saved best checkpoint: {checkpoint_path}")
        else:
            self.logger.info(f"  Saved checkpoint: {checkpoint_path}")
