import hydra
import torch
import torch.nn.functional as F
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from semcom.channel.visualize import plot_channel_effects
from semcom.tracking.run_naming import make_run_name
from semcom.tracking.seed import set_seed
from semcom.tracking.wandb_logger import init_run


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    set_seed(cfg.seed)

    dataset = hydra.utils.instantiate(cfg.dataset, split="train")
    loader = DataLoader(
        dataset, batch_size=cfg.experiment.batch_size, shuffle=True, num_workers=cfg.experiment.num_workers
    )
    batch = next(iter(loader))

    x = batch["input"]
    if not torch.is_tensor(x):
        raise TypeError(
            f"Smoke test needs a tensor-valued 'input' to push through the channel; "
            f"got {type(x)} (expected for text datasets — skip the channel step for those)."
        )
    x = x.flatten(start_dim=1).float()
    if x.shape[-1] % 2 != 0:
        x = F.pad(x, (0, 1))

    channel = hydra.utils.instantiate(cfg.channel)
    choices = HydraConfig.get().runtime.choices
    run_name = make_run_name(
        dataset.modality, choices["dataset"], choices["channel"], snr_db=cfg.experiment.snr_range_db[1], seed=cfg.seed
    )
    run = init_run(cfg, run_name, project=cfg.tracking.project, mode=cfg.tracking.mode)

    snr_lo, snr_hi = cfg.experiment.snr_range_db
    snr = snr_lo
    while snr <= snr_hi:
        y = channel(x, float(snr))
        mse = F.mse_loss(y, x).item()
        run.log({"snr_db": snr, "mse": mse})
        snr += cfg.experiment.snr_step_db

    fig = plot_channel_effects()
    run.log({"channel_sanity_check": fig})
    run.finish()

    print(f"Smoke test complete: {choices['dataset']} dataset over {choices['channel']} channel.")


if __name__ == "__main__":
    main()
