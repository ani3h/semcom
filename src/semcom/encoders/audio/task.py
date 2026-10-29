import torch
import torch.nn.functional as F

from ..base import TaskSpec
from .metrics import log_spectral_distance, pesq_mean, si_sdr, stoi_mean
from .model import SEAudioTransceiver


class AudioReconTask(TaskSpec):
    """Speech reconstruction on the Edinburgh (Valentini) clean-speech corpus.

    Chosen over a classification task because the corpus is parallel
    clean/noisy speech with transcripts and no class or keyword labels."""

    modality = "audio"
    primary_metric = "stoi"  # event audio (use_stoi false) is judged on si_sdr / lsd instead

    def __init__(self, model: dict | None = None, eval: dict | None = None, sample_rate: int = 16000):
        self.model_cfg = dict(model or {})
        eval = dict(eval or {})
        self.max_items = eval.get("max_items", 300)
        self.eval_batch_size = eval.get("batch_size", 50)
        self.eval_seed = eval.get("seed", 1234)
        self.use_pesq = eval.get("pesq", True)
        self.use_stoi = eval.get("stoi", True)  # STOI is a speech-intelligibility measure: switch it off for event audio
        self.use_lsd = eval.get("lsd", False)
        self.fit_items = eval.get("scale_fit_items", 500)
        self.sample_rate = sample_rate
        self.scale = 1.0
        self.metric_names = ["si_sdr"] + (["stoi"] if self.use_stoi else []) + (["pesq"] if self.use_pesq else []) \
            + (["lsd"] if self.use_lsd else [])

    def fit(self, train_dataset) -> None:
        n = min(self.fit_items, len(train_dataset))
        waves = torch.stack([train_dataset[i]["input"] for i in range(n)])
        self.scale = 1.0 / waves.pow(2).mean().sqrt().item()

    def state_dict(self) -> dict:
        return {"scale": self.scale}

    def load_state_dict(self, state: dict) -> None:
        self.scale = state["scale"]

    def build_model(self) -> SEAudioTransceiver:
        return SEAudioTransceiver(scale=self.scale, **self.model_cfg)

    def collate(self, items):
        return {"x": torch.stack([it["input"] for it in items])}

    def loss(self, model, batch, channel, snr_db):
        out = model(batch["x"], channel, snr_db)
        return F.mse_loss(out * model.scale, batch["x"] * model.scale)

    @torch.no_grad()
    def evaluate(self, model, dataset, channel, snr_db, device) -> dict[str, float]:
        model.eval()
        torch.manual_seed(self.eval_seed + int(round(snr_db * 10)))
        n = min(self.max_items, len(dataset))
        est, ref = [], []
        for start in range(0, n, self.eval_batch_size):
            items = [dataset[i] for i in range(start, min(start + self.eval_batch_size, n))]
            batch = self.to_device(self.collate(items), device)
            est.append(model(batch["x"], channel, snr_db).cpu())
            ref.append(batch["x"].cpu())
        est, ref = torch.cat(est), torch.cat(ref)

        results = {"si_sdr": si_sdr(est, ref).mean().item()}
        if self.use_stoi:
            results["stoi"] = stoi_mean(est, ref, self.sample_rate)
        if self.use_lsd:
            results["lsd"] = log_spectral_distance(est, ref).mean().item()
        if self.use_pesq:
            score = pesq_mean(est, ref, self.sample_rate)
            results["pesq"] = float("nan") if score is None else score
        return results
