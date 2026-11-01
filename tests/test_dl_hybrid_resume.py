"""Checkpoint restart must reproduce an uninterrupted CPU training trajectory."""
import copy

import numpy as np
import pytest
import torch

from src import conv_bigru_attention as network


def tiny_fit_arguments():
    rng = np.random.default_rng(12)
    tokens = np.array([[2, 2, 2, 0], [2, 2, 0, 0]] * 3, dtype=np.int64)
    arrays = (tokens, tokens.copy(), rng.normal(size=(6, 4, 2)).astype(np.float32),
              rng.normal(size=(6, 4, 4)).astype(np.float32),
              rng.normal(size=(6, 5)).astype(np.float32))
    y = np.array([0, 1, 2, 0, 1, 2])
    classes = np.array(["a", "b", "c"])
    return dict(train_arrays=arrays, y=y, eval_arrays=arrays, eval_y=y, classes=classes,
                eval_browsers=["Firefox"] * 6, browser_map=dict.fromkeys(classes, "Firefox"),
                num_tokens=3, num_ctrls=3, epochs=3, batch_size=4, diagnostic_best=True)


def test_epoch_resume_restores_optimizer_scheduler_rng_and_averaged_weights(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    arguments = tiny_fit_arguments()
    uninterrupted, full_history = network.train_network(**arguments)
    expected = copy.deepcopy(uninterrupted.state_dict())
    path = tmp_path / "training.pt"
    original_save = network.save_training_checkpoint

    class SimulatedStop(Exception):
        pass

    def stop_after_saved_epoch(payload, destination):
        original_save(payload, destination)
        if payload["epoch"] == 1:
            raise SimulatedStop()

    monkeypatch.setattr(network, "save_training_checkpoint", stop_after_saved_epoch)
    with pytest.raises(SimulatedStop):
        network.train_network(**arguments, checkpoint_path=path)
    saved = torch.load(path, map_location="cpu", weights_only=False)
    assert saved["epoch"] == 1
    assert saved["optimizer_state"]["state"]
    assert saved["scheduler_state"]["last_epoch"] == 1
    monkeypatch.setattr(network, "save_training_checkpoint", original_save)
    resumed, resumed_history = network.train_network(**arguments, checkpoint_path=path)
    assert resumed_history["curve"] == full_history["curve"]
    for name in expected:
        torch.testing.assert_close(expected[name], resumed.state_dict()[name], rtol=0, atol=0)
    assert torch.load(path, map_location="cpu", weights_only=False)["epoch"] == 3
    # Reopening a complete epoch checkpoint only repeats deterministic finalization.
    finalized, _ = network.train_network(**arguments, checkpoint_path=path)
    for name in expected:
        torch.testing.assert_close(expected[name], finalized.state_dict()[name], rtol=0, atol=0)


def test_checkpoint_rejects_changed_training_data(tmp_path):
    torch.set_num_threads(1)
    arguments = tiny_fit_arguments()
    path = tmp_path / "training.pt"
    network.train_network(**arguments, checkpoint_path=path)
    altered = [x.copy() for x in arguments["train_arrays"]]
    altered[-1][0, 0] += 1
    arguments["train_arrays"] = tuple(altered)
    with pytest.raises(ValueError, match="Checkpoint does not match"):
        network.train_network(**arguments, checkpoint_path=path)
