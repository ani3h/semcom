"""Stage 2 embedding builders: transmit a fixed item set through a frozen transceiver and a channel, and embed what comes out in S.

Approach A: head(received symbols).  Approach B: pretrained CLIP/CLAP on the decoded output.  Both use the SAME channel realisations: the
noise stream is seeded per (base seed, draw, SNR, chunk), so a given item sees the identical noise under A, B and every head seed (a paired
design). The clean symbols are computed once per chunk; each (draw, SNR) then costs only the channel plus the head / decoder.
"""
import torch
import torch.nn.functional as F

from .fidelity import derangement

SNR_GRID = [0.0, 5.0, 10.0, 15.0, 20.0, 25.0]
DRAWS = 5
CHUNK = {"cifar10": 500, "mnist": 500, "text": 250, "caption_text": 250, "audio": 250, "esc50_audio": 100, "prompts": 100}
# first-N items of each split used for evaluation (a seeded shuffle for text; arbitrary-order test sets for the rest)
EVAL_ITEMS = {"val": {"cifar10": 1000, "mnist": 1000, "text": 1000, "caption_text": 1000, "audio": 1000, "esc50_audio": 400},
              "test": {"cifar10": 2000, "mnist": 2000, "text": 1000, "caption_text": 1000, "audio": 824, "esc50_audio": 400}}  # speech: all 824, the Week 5 gallery size


def chunk_seed(base_seed: int, draw: int, snr: float, chunk: int) -> int:
    return base_seed * 1_000_003 + draw * 10_007 + int(snr) * 101 + chunk


@torch.no_grad()
def clean_symbols(w4, adapter, items: list[dict], device, chunk: int) -> list[dict]:
    """Per chunk: the transceiver's clean symbols plus the side information the head and the decoder need."""
    out = []
    for s in range(0, len(items), chunk):
        batch = w4.spec.to_device(w4.spec.collate(items[s:s + chunk]), device)
        out.append({"sym": w4.model.encode(adapter.tx_input(batch)), "head_side": adapter.head_side(batch),
                    "decode_side": adapter.decode_side(batch)})
    return out


def _empty(n_items: int, snrs, draws: int, dim: int = 512):
    return torch.empty(draws, len(snrs), n_items, dim, dtype=torch.float16)


@torch.no_grad()
def embed_A(head, domain, target: str, chunks: list[dict], channel, anchors_raw: torch.Tensor | None, snrs=SNR_GRID, draws: int = DRAWS,
            base_seed: int = 0) -> dict:
    """Head embeddings in S for every (draw, SNR), plus the noise-free embedding.
    target "raw": the head outputs the raw CLIP/CLAP space, mapped into S by the domain; also returns the raw cosine to `anchors_raw` for the
    own anchor and for a fixed wrong anchor (the derangement used everywhere), i.e. the raw-cosine baseline in the head's own training space.
    target "centered": the head outputs S directly (its raw cosine is its centered cosine, recomputed from S downstream)."""
    n = sum(c["sym"].size(0) for c in chunks)
    z = _empty(n, snrs, draws)
    raw = anchors_raw is not None
    raw_cos = torch.empty(draws, len(snrs), n) if raw else None
    raw_mis = torch.empty(draws, len(snrs), n) if raw else None
    to_S = domain.to_S if target == "raw" else (lambda x: x)
    a_raw = anchors_raw.to(chunks[0]["sym"].device) if raw else None
    perm = derangement(n).to(chunks[0]["sym"].device)

    def one(rx, c):
        zr = head(rx, **c["head_side"])
        return to_S(zr), zr

    def raw_scores(zr, pos):
        m = zr.size(0)
        return (zr * a_raw[pos:pos + m]).sum(-1).cpu(), (zr * a_raw[perm[pos:pos + m]]).sum(-1).cpu()

    for d in range(draws):
        for si, snr in enumerate(snrs):
            pos = 0
            for ci, c in enumerate(chunks):
                torch.manual_seed(chunk_seed(base_seed, d, snr, ci))
                zs, zr = one(channel(c["sym"], snr), c)
                m = zs.size(0)
                z[d, si, pos:pos + m] = zs.half().cpu()
                if raw:
                    raw_cos[d, si, pos:pos + m], raw_mis[d, si, pos:pos + m] = raw_scores(zr, pos)
                pos += m
    zc, pos = torch.empty(n, 512, dtype=torch.float16), 0
    rc = torch.empty(n) if raw else None
    for c in chunks:
        zs, zr = one(c["sym"], c)
        m = zs.size(0)
        zc[pos:pos + m] = zs.half().cpu()
        if raw:
            rc[pos:pos + m] = raw_scores(zr, pos)[0]
        pos += m
    return {"z": z, "z_clean": zc, "raw_cos": raw_cos, "raw_cos_mis": raw_mis, "raw_cos_clean": rc}


@torch.no_grad()
def embed_B(w4, adapter, domain, chunks: list[dict], channel, snrs=SNR_GRID, draws: int = DRAWS, base_seed: int = 0, audio_workers: int = 0,
            progress=None) -> dict:
    """Pretrained-encoder embeddings (CLIP / CLAP) of the DECODED output, mapped into S, for every (draw, SNR) and for the noise-free decode."""
    n = sum(c["sym"].size(0) for c in chunks)
    z = _empty(n, snrs, draws)

    def embed(rx, c):
        out = w4.model.decode(rx, **c["decode_side"])
        if audio_workers and adapter.modality == "audio":
            e = adapter.anchors.encode_audio(out, adapter.sample_rate, num_workers=audio_workers)
        else:
            e = adapter.embed_decoded(out)
        return domain.to_S(F.normalize(e.float().to(rx.device), dim=-1))

    for d in range(draws):
        for si, snr in enumerate(snrs):
            pos = 0
            for ci, c in enumerate(chunks):
                torch.manual_seed(chunk_seed(base_seed, d, snr, ci))
                zs = embed(channel(c["sym"], snr), c)
                z[d, si, pos:pos + zs.size(0)] = zs.half().cpu()
                pos += zs.size(0)
            if progress:
                progress(d, snr)
    zc, pos = torch.empty(n, 512, dtype=torch.float16), 0
    for c in chunks:
        zs = embed(c["sym"], c)
        zc[pos:pos + zs.size(0)] = zs.half().cpu()
        pos += zs.size(0)
    return {"z": z, "z_clean": zc}
