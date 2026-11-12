import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient

from skimmer.env import OBS_DIM
from skimmer.ppo import ActorCritic


@pytest.fixture()
def client(tmp_path, monkeypatch):
    path = tmp_path / "p.pt"
    torch.save(ActorCritic().state_dict(), path)
    monkeypatch.setenv("POLICY_PATH", str(path))
    import skimmer.serve as serve

    serve._net = None
    return TestClient(serve.app)


def test_act_returns_valid_action(client):
    r = client.post("/act", json={"obs": [0.0] * OBS_DIM})
    assert r.status_code == 200
    assert 0 <= r.json()["action"] < 9


def test_wrong_obs_length_rejected(client):
    assert client.post("/act", json={"obs": [0.0] * 3}).status_code == 422
