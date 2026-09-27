import numpy as np
import pytest
import torch
from scripts.layer_study.spectral import feature_spectrum, filter_patch_tokens, frequency_radius


def wave(k, amplitude=1.0):
    signal = amplitude * np.cos(2 * np.pi * k * np.arange(14) / 14)
    return np.tile(signal, (14, 1))[None]


@pytest.mark.parametrize("frequency,band", [(1, "low"), (3, "mid"), (5, "high"), (7, "high")])
def test_known_fourier_mode_is_in_correct_band(frequency, band):
    spectrum = feature_spectrum(wave(frequency), "rectangular")
    assert spectrum["bands"][band] == pytest.approx(1, abs=1e-12)
    assert sum(spectrum["profile"]) == pytest.approx(1)
    assert spectrum["ac_energy"] == pytest.approx(spectrum["windowed_spatial_energy"])


def test_hann_dc_removed_and_amplitude_invariant():
    x = wave(3) + 0.5 * wave(5)
    a = feature_spectrum(x)
    b = feature_spectrum(8 * x + 23)
    np.testing.assert_allclose(a["profile"], b["profile"], atol=1e-12)
    assert sum(a["bands"].values()) == pytest.approx(1)
    assert a["ac_energy"] == pytest.approx(a["windowed_spatial_energy"], rel=1e-12)


def test_constant_feature_has_no_ac_energy():
    assert not feature_spectrum(np.ones((3, 14, 14)))["active"]


def test_channels_do_not_cancel_and_diagonal_nyquist_retained():
    x = (-1.0) ** np.indices((14, 14)).sum(axis=0)
    spectrum = feature_spectrum(np.stack([x, -x]), "rectangular")
    assert spectrum["bands"]["high"] == pytest.approx(1)
    assert spectrum["profile"][-1] == pytest.approx(1)
    assert sum(spectrum["mode_counts"]) == 195
    assert frequency_radius(14, 14).max() == pytest.approx(np.sqrt(98))


def test_rms_matched_filter_preserves_cls_mean_and_energy():
    torch.manual_seed(4)
    tokens = torch.randn(2, 197, 8)
    result = filter_patch_tokens(tokens, "lowpass", match_rms=True)
    assert torch.equal(tokens[:, :1], result[:, :1])
    original = tokens[:, 1:]
    filtered = result[:, 1:]
    torch.testing.assert_close(original.mean(1), filtered.mean(1), atol=1e-6, rtol=1e-5)
    energy_before = ((original - original.mean(1, keepdim=True)) ** 2).sum((1, 2))
    energy_after = ((filtered - filtered.mean(1, keepdim=True)) ** 2).sum((1, 2))
    torch.testing.assert_close(energy_before, energy_after, atol=1e-4, rtol=1e-5)
    torch.testing.assert_close(filter_patch_tokens(tokens, "sham"), tokens, atol=1e-6, rtol=1e-5)


def test_lowpass_removes_high_mode_but_preserves_low_mode():
    x = wave(1) + wave(5)
    tokens = torch.zeros(1, 197, 1)
    tokens[:, 1:, 0] = torch.tensor(x.reshape(1, -1), dtype=torch.float32)
    result = filter_patch_tokens(tokens, "lowpass")
    np.testing.assert_allclose(result[0, 1:, 0].numpy().reshape(14, 14), wave(1)[0], atol=1e-6)
