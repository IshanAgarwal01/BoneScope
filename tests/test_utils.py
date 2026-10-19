import torch

from mds.utils import CSVLogger, amp_enabled, get_device, seed_everything


def test_seed_makes_torch_repeatable():
    seed_everything(0)
    a = torch.rand(3)
    seed_everything(0)
    assert torch.equal(a, torch.rand(3))


def test_amp_only_on_cuda():
    assert amp_enabled(torch.device("cpu"), True) is False
    assert amp_enabled(torch.device("cuda"), True) is True
    assert amp_enabled(torch.device("cuda"), False) is False


def test_device_fallback_returns_a_device():
    assert get_device("auto").type in {"cuda", "mps", "cpu"}
    assert get_device("cpu").type == "cpu"


def test_csv_logger(tmp_path):
    log = CSVLogger(tmp_path / "log.csv")
    log.log({"epoch": 0, "val_macro_f1": 0.5})
    log.log({"epoch": 1, "val_macro_f1": 0.6})
    lines = (tmp_path / "log.csv").read_text().strip().splitlines()
    assert lines[0] == "epoch,val_macro_f1" and len(lines) == 3
