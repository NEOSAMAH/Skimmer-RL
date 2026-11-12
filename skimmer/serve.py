"""Serve a trained policy over HTTP: POST an 18-d observation, get back a steering action.

    POLICY_PATH=results/policy_dr_seed0.pt uvicorn skimmer.serve:app
"""
from __future__ import annotations

import os
import time

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .env import OBS_DIM
from .ppo import ActorCritic

ACTION_NAMES = ["E", "NE", "N", "NW", "W", "SW", "S", "SE", "HOLD"]
app = FastAPI(title="Skimmer policy")
_net: ActorCritic | None = None


def get_net() -> ActorCritic:
    global _net
    if _net is None:
        _net = ActorCritic()
        _net.load_state_dict(torch.load(os.environ.get("POLICY_PATH", "results/policy_seed0.pt")))
        _net.eval()
    return _net


class Observation(BaseModel):
    obs: list[float]


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/act")
def act(body: Observation):
    if len(body.obs) != OBS_DIM:
        raise HTTPException(422, f"obs must have {OBS_DIM} values, got {len(body.obs)}")
    t0 = time.perf_counter()
    with torch.no_grad():
        dist, value = get_net()(torch.tensor([body.obs], dtype=torch.float32))
    a = int(dist.probs.argmax(-1))
    return {
        "action": a,
        "direction": ACTION_NAMES[a],
        "confidence": float(dist.probs[0, a]),
        "state_value": float(value[0]),
        "latency_ms": round((time.perf_counter() - t0) * 1000, 3),
    }
