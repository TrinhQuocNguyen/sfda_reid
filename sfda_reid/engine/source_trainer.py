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

    def train(self):
        for epoch in range(self.cfg.source.num_epochs):
            self.model.train()
            total_loss, total_ce, total_supcon = 0, 0, 0
            for batch in self.train_loader:
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
            self.scheduler.step()
            avg_loss = total_loss / len(self.train_loader)
            avg_ce = total_ce / len(self.train_loader)
            avg_supcon = total_supcon / len(self.train_loader)
            self.writer.add_scalar('Loss/total', avg_loss, epoch)
            self.writer.add_scalar('Loss/ce', avg_ce, epoch)
            self.writer.add_scalar('Loss/supcon', avg_supcon, epoch)
            self.writer.add_scalar('LR', self.optimizer.param_groups[0]['lr'], epoch)
            if (epoch + 1) % self.cfg.eval_every == 0:
                rank1 = self.evaluate(epoch)
                if rank1 > self.best_rank1:
                    self.best_rank1 = rank1
                    self.save_checkpoint(epoch, best=True)
            if (epoch + 1) % self.cfg.log_every == 0:
                self.logger.info(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, CE={avg_ce:.4f}, SupCon={avg_supcon:.4f}")
        self.writer.close()

    def evaluate(self, epoch: int) -> float:
        self.model.eval()
        from ..engine.evaluator import ReIDEvaluator
        evaluator = ReIDEvaluator()
        metrics = evaluator.evaluate(self.model, self.val_loader, self.val_loader)
        rank1 = metrics['rank1']
        self.writer.add_scalar('Rank1', rank1, epoch)
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
        torch.save(state, f"{out_dir}/{fname}")
