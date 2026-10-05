from hydra import compose, initialize


def test_config_composes_with_defaults():
    with initialize(version_base=None, config_path="../configs"):
        cfg = compose(config_name="config")
    assert cfg.seed == 42
    assert cfg.dataset._target_ == "semcom.data.image_cifar10.CIFAR10Dataset"
    assert cfg.channel._target_ == "semcom.channel.awgn.AWGNChannel"
    assert cfg.experiment.snr_range_db == [0, 25]


def test_config_overrides_dataset_and_channel():
    with initialize(version_base=None, config_path="../configs"):
        cfg = compose(config_name="config", overrides=["dataset=mnist", "channel=rayleigh"])
    assert cfg.dataset._target_ == "semcom.data.image_mnist.MNISTDataset"
    assert cfg.channel._target_ == "semcom.channel.rayleigh.RayleighChannel"


def test_data_root_interpolation_reaches_dataset_config():
    with initialize(version_base=None, config_path="../configs"):
        cfg = compose(config_name="config", overrides=["data_root=/tmp/custom_cache"])
    assert cfg.dataset.root == "/tmp/custom_cache"
