"""ROCm/torch GPU inference worker for the forest_change plugin.

Runs in the ROCm PyTorch venv (D:\\comfyui\\...\\venv), started by core.RocmModel
as a persistent subprocess. Protocol (stdin/stdout, one line per request):

  plugin -> <path to .npz with pre (N,6,128,128), post, seq (N,4,9) float32>
  worker -> OK <path>   (writes <path>.out.npz with logits (N,4) float32)
            ERR <message>

On start (after the model is loaded) the worker prints READY.
"""

import os
import sys

import numpy as np
import torch
import torch.nn as nn

SRC = os.environ.get("FC_SRC", r"C:\Users\artyom\Desktop\лесной хакатон\src")
SAVE = os.environ.get("FC_WEIGHTS", r"D:\forest_hack_data\dofa_weights")
sys.path.insert(0, SRC)

from cfc import CfCClassifier  # noqa: E402
from train_dofa_ft import Net, WL, load_dofa  # noqa: E402
from train_fusion_big4 import Head  # noqa: E402

NC = 4
DT = [0.0, 0.72, 1.15, 1.0]
CHUNK = 16


class Fusion(nn.Module):
    def __init__(self, net, cfc, comb):
        super().__init__()
        self.net, self.cfc, self.comb = net, cfc, comb
        self.register_buffer("wl", torch.tensor(WL, dtype=torch.float32))
        self.register_buffer("dt", torch.tensor(DT, dtype=torch.float32).unsqueeze(0))

    def forward(self, pre, post, seq):
        s_log = self.net.head(torch.cat([self.net._feats(pre, self.wl),
                                         self.net._feats(post, self.wl)], 1))
        t_log = self.cfc(seq, self.dt.expand(seq.shape[0], -1))
        return self.comb(None, s_log, None, t_log)


def build(dev):
    net = Net(load_dofa(dev), freeze_upto=0)
    net.load_state_dict(torch.load(os.path.join(SAVE, "dofa_vozh.pth"), map_location="cpu"))
    cfc = CfCClassifier(9, hidden_size=128, num_classes=NC, dropout=0.2,
                        backbone_units=128, backbone_layers=2, pool="mean")
    cfc.load_state_dict(torch.load(os.path.join(SAVE, "cfc_vozh.pth"), map_location="cpu"))
    comb = Head("logit")
    comb.load_state_dict(torch.load(os.path.join(SAVE, "fusion_vozh.pth"), map_location="cpu"))
    return Fusion(net, cfc, comb).to(dev).eval()


def infer(model, dev, path):
    z = np.load(path)
    pre = torch.from_numpy(np.ascontiguousarray(z["pre"], dtype=np.float32))
    post = torch.from_numpy(np.ascontiguousarray(z["post"], dtype=np.float32))
    seq = torch.from_numpy(np.ascontiguousarray(z["seq"], dtype=np.float32))
    outs = []
    with torch.no_grad():
        for i in range(0, len(pre), CHUNK):
            outs.append(model(pre[i:i + CHUNK].to(dev), post[i:i + CHUNK].to(dev),
                              seq[i:i + CHUNK].to(dev)).float().cpu().numpy())
    return np.concatenate(outs) if outs else np.zeros((0, NC), np.float32)


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build(dev)
    print(f"READY device={dev} name={torch.cuda.get_device_name(0) if dev.type == 'cuda' else 'cpu'}",
          flush=True)
    for line in sys.stdin:
        path = line.strip()
        if not path:
            continue
        if path in ("exit", "quit"):
            break
        try:
            out = infer(model, dev, path)
            np.savez(path + ".out.npz", logits=out)
            print(f"OK {path}", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"ERR {type(exc).__name__}: {str(exc)[:200]}", flush=True)


if __name__ == "__main__":
    main()
