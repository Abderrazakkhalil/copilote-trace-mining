import pytest
from src.parser import (
    _split_glued_tokens,
    _parse_action_token,
    _compute_window_counts,
    parse_session,
    RE_TIME,
    RE_ACTION,
)


def test_re_time():
    assert RE_TIME.match("t0")
    assert RE_TIME.match("t5")
    assert RE_TIME.match("t12345")
    assert not RE_TIME.match("t")
    assert not RE_TIME.match("action_t5")


def test_split_glued_tokens():
    raw = ["Création d'un écran(AccueilController)t5", "t10", "Saisie dans un champ"]
    res = _split_glued_tokens(raw)
    assert res == ["Création d'un écran(AccueilController)", "t5", "t10", "Saisie dans un champ"]


def test_parse_action_token_standard():
    tok = "Exécution d'un bouton(ValiderCtrl)<Config1>$DomaineAdmin$1"
    res = _parse_action_token(tok)
    assert res["base_action"] == "Exécution d'un bouton"
    assert res["controller"] == "ValiderCtrl"
    assert res["config"] == "Config1"
    assert res["domain"] == "DomaineAdmin"
    assert res["suffix_one"] is True


def test_parse_action_token_minimal():
    tok = "Saisie dans un champ"
    res = _parse_action_token(tok)
    assert res["base_action"] == "Saisie dans un champ"
    assert res["controller"] is None
    assert res["config"] is None
    assert res["domain"] is None
    assert res["suffix_one"] is False


def test_compute_window_counts():
    time_markers = [5, 10, 15]
    n_actions_between = [2, 4, 1]
    counts = _compute_window_counts(time_markers, n_actions_between, window=5)
    # Total actions = 7, duration 15s -> 4 windows (0, 5, 10, 15)
    assert sum(counts) == 7


def test_parse_session():
    tokens = [
        "Firefox",
        "t0",
        "Exécution d'un bouton(BtnCtrl)",
        "t5",
        "Saisie dans un champ",
        "t10",
    ]
    res = parse_session(tokens, user="user_123")
    assert res["user"] == "user_123"
    assert res["browser"] == "Firefox"
    assert len(res["raw_actions"]) == 2
    assert res["base_actions"] == ["Exécution d'un bouton", "Saisie dans un champ"]
    assert res["controllers"] == ["BtnCtrl", None]
    assert res["time_markers"] == [0, 5, 10]
