"""Client script for the step-1 sanity check: official NVFLARE FedAvgRecipe +
Client API, real PyTorch training of the paper CNN on a CIFAR-10 shard."""
import os
import sys

import torch
import torch.nn.functional as F

import nvflare.client as flare

ROOT = os.environ["FEDASMU_ROOT"]
sys.path.insert(0, ROOT)
from src.datasets.cifar import Normalizer, load_tensor_dataset  # noqa: E402
from src.models import build_model  # noqa: E402


def main():
    flare.init()
    site = flare.get_site_name()
    idx = int(site.split("-")[-1]) - 1
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    x, y, xt, yt, n = load_tensor_dataset("cifar10")
    g = torch.Generator().manual_seed(idx)
    sel = torch.randperm(len(x), generator=g)[:2000]
    x, y = x[sel].to(dev), y[sel].to(dev)
    xt, yt = xt[:2000].to(dev), yt[:2000].to(dev)
    norm = Normalizer("cifar10", dev)
    model = build_model("cnn", n).to(dev)
    while flare.is_running():
        inp = flare.receive()
        model.load_state_dict({k: torch.as_tensor(v) for k, v in inp.params.items()})
        with torch.no_grad():
            acc0 = (model(norm(xt)).argmax(1) == yt).float().mean().item()
        opt = torch.optim.SGD(model.parameters(), lr=0.03)
        model.train()
        for ep in range(2):
            perm = torch.randperm(len(x), device=dev)
            for i in range(0, len(x), 64):
                b = perm[i:i + 64]
                loss = F.cross_entropy(model(norm(x[b])), y[b])
                opt.zero_grad(); loss.backward(); opt.step()
        print(f"[{site}] round={inp.current_round} recv_global_acc={acc0:.4f} last_loss={loss.item():.4f}", flush=True)
        flare.send(flare.FLModel(params={k: v.cpu() for k, v in model.state_dict().items()},
                                 metrics={"accuracy": acc0}, meta={"NUM_STEPS_CURRENT_ROUND": 64}))


if __name__ == "__main__":
    main()
