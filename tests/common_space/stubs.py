"""Download-free stand-ins for frozen CLIP/CLAP: fixed random projections with the same encode_* API."""
import torch
import torch.nn.functional as F


class StubAnchors:
    def __init__(self, dim: int = 512, seed: int = 0, text_device="cpu"):
        self.text_device = text_device
        g = torch.Generator().manual_seed(seed)
        self.w_img = torch.randn(3 * 8 * 8, dim, generator=g)
        self.w_txt = torch.randn(256, dim, generator=g)
        self.w_aud = torch.randn(32, dim, generator=g)

    @staticmethod
    def _out(feat, w, device):
        return F.normalize(feat @ w + 1e-3, dim=-1).to(device)

    def encode_images(self, x):
        device = x.device
        x = x.detach().float().cpu()
        if x.size(1) == 1:
            x = x.expand(-1, 3, -1, -1)
        return self._out(F.adaptive_avg_pool2d(x, 8).flatten(1), self.w_img, device)

    def encode_text(self, texts):
        counts = torch.zeros(len(texts), 256)
        for i, t in enumerate(texts):
            for ch in t.encode("utf-8", "ignore"):
                counts[i, ch] += 1
        return self._out(counts, self.w_txt, self.text_device)

    def encode_audio(self, wave, sample_rate: int = 16000):
        device = wave.device
        w = wave.detach().float().cpu()
        n = (w.size(1) // 32) * 32
        return self._out(w[:, :n].view(w.size(0), 32, -1).abs().mean(-1), self.w_aud, device)
