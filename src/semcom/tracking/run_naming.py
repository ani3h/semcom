from datetime import datetime, timezone


def make_run_name(modality: str, dataset: str, channel: str, snr_db: float | str, seed: int) -> str:
    """{modality}-{dataset}-{channel}-snr{snr}-seed{seed}-{UTC timestamp}.
    snr_db may be a number (single-SNR run) or a label such as "U0-25" for
    training with SNR drawn uniformly from a range."""
    snr = snr_db if isinstance(snr_db, str) else f"{snr_db:g}"
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{modality}-{dataset}-{channel}-snr{snr}-seed{seed}-{ts}"
