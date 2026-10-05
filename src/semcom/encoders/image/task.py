import torch
import torch.nn.functional as F
from torchmetrics.image import StructuralSimilarityIndexMeasure

from ..base import TaskSpec
from .model import JSCCClassifier, JSCCImage


def psnr(x: torch.Tensor, y: torch.Tensor, data_range: float = 1.0) -> torch.Tensor:
    """Per-image PSNR in dB for (B, C, H, W) tensors."""
    mse = (x - y).pow(2).flatten(1).mean(dim=1).clamp_min(1e-10)
    return 10 * torch.log10(data_range**2 / mse)


def _stack(items: list[dict]) -> dict[str, torch.Tensor]:
    return {"x": torch.stack([it["input"] for it in items]), "y": torch.stack([it["label"] for it in items])}


class _ImageTask(TaskSpec):
    modality = "image"

    def __init__(self, model: dict | None = None, eval: dict | None = None):
        self.model_cfg = dict(model or {})
        eval = dict(eval or {})
        self.max_items = eval.get("max_items", None)
        self.eval_batch_size = eval.get("batch_size", 500)
        self.eval_seed = eval.get("seed", 1234)

    def collate(self, items):
        return _stack(items)

    def _eval_batches(self, dataset, device):
        n = len(dataset) if self.max_items is None else min(self.max_items, len(dataset))
        for start in range(0, n, self.eval_batch_size):
            items = [dataset[i] for i in range(start, min(start + self.eval_batch_size, n))]
            yield self.to_device(self.collate(items), device)


class CIFARReconTask(_ImageTask):
    """CIFAR-10 image transmission: reconstruct the image from noisy symbols."""

    metric_names = ["psnr", "ssim"]
    primary_metric = "psnr"

    def build_model(self) -> JSCCImage:
        return JSCCImage(**self.model_cfg)

    def loss(self, model, batch, channel, snr_db):
        return F.mse_loss(model(batch["x"], channel, snr_db), batch["x"])

    @torch.no_grad()
    def evaluate(self, model, dataset, channel, snr_db, device) -> dict[str, float]:
        model.eval()
        torch.manual_seed(self.eval_seed + int(round(snr_db * 10)))
        ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
        psnr_sum, count = 0.0, 0
        for batch in self._eval_batches(dataset, device):
            out = model(batch["x"], channel, snr_db).clamp(0, 1).cpu()
            x = batch["x"].cpu()
            psnr_sum += psnr(out, x).sum().item()
            count += x.size(0)
            ssim.update(out, x)
        return {"psnr": psnr_sum / count, "ssim": ssim.compute().item()}


class MNISTClassTask(_ImageTask):
    """MNIST classification straight from the received symbols."""

    metric_names = ["accuracy"]
    primary_metric = "accuracy"

    def build_model(self) -> JSCCClassifier:
        return JSCCClassifier(**self.model_cfg)

    def loss(self, model, batch, channel, snr_db):
        return F.cross_entropy(model(batch["x"], channel, snr_db), batch["y"])

    @torch.no_grad()
    def evaluate(self, model, dataset, channel, snr_db, device) -> dict[str, float]:
        model.eval()
        torch.manual_seed(self.eval_seed + int(round(snr_db * 10)))
        correct, count = 0, 0
        for batch in self._eval_batches(dataset, device):
            pred = model(batch["x"], channel, snr_db).argmax(-1)
            correct += (pred == batch["y"]).sum().item()
            count += batch["y"].numel()
        return {"accuracy": correct / count}
