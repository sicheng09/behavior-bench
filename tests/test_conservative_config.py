import pytest

from pufferlib.conservative.config import parse_conservative_config


def test_defaults_for_phase_a():
    cfg = parse_conservative_config({})
    assert cfg.partner_mode == "action_constraint"
    assert cfg.partner_max_abs_steer == pytest.approx(0.333)
    assert cfg.partner_constrain_accel is False
    assert cfg.partner_target_headway == pytest.approx(1.8)
    assert cfg.w_center == pytest.approx(0.05)
    assert cfg.w_align == pytest.approx(0.05)
    assert cfg.w_steer == pytest.approx(0.05)
    assert cfg.w_gap == pytest.approx(0.10)


def test_rejects_unknown_mode():
    with pytest.raises(ValueError, match="partner_mode"):
        parse_conservative_config({"partner_mode": "idm"})


def test_rejects_nonpositive_headway():
    with pytest.raises(ValueError, match="headway"):
        parse_conservative_config({"partner_target_headway": 0.0})


def test_bool_parsing_from_ini_strings():
    cfg = parse_conservative_config(
        {"partner_constrain_accel": "False", "partner_mode": "both"}
    )
    assert cfg.partner_constrain_accel is False
    assert cfg.partner_mode == "both"
