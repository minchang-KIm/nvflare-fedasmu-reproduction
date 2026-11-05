"""FedBuff aggregation built on NVFLARE 2.9.0's official ``BuffModelManager``.

* ``official``: ``nvflare.edge.assessors.buff_model_manager.BuffModelManager`` used
  unchanged.  Its update is  w <- w + global_lr * (sum_v s_v * sum(diffs_v)) / N  with
  s_v = 1/(1 + (V_new - v)^0.5) where V_new is the version *being generated*
  (i.e. staleness+1) when ``staleness_weight=True``, else 1.
* ``paper``: identical buffering/normalisation, but the FedBuff paper's staleness
  scaling s(tau) = 1/sqrt(1 + tau), tau = V_cur - v (Nguyen et al. 2022, Sec. 5).
Both: a new model is produced once >= K device updates were buffered (counted per
device), updates are model *differences* (w_local - w_received), the average is over
the number of buffered updates (paper Algorithm 1 line 11: Delta/K).
"""
import numpy as np

from nvflare.apis.dxo import DXO, DataKind
from nvflare.app_common.abstract.model import make_model_learnable
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.app_event_type import AppEventType
from nvflare.edge.aggregators.model_update_dxo import ModelUpdateDXOAggregator
from nvflare.edge.assessors.buff_model_manager import BuffModelManager, _ModelState
from nvflare.edge.mud import ModelUpdate


class PaperFedBuffModelManager(BuffModelManager):
    """BuffModelManager with the FedBuff-paper staleness function."""

    def generate_new_model(self, fl_ctx) -> None:
        if self.current_model_version == 0:
            return super().generate_new_model(fl_ctx)
        cur = self.current_model_version
        acc, num = {}, 0
        for v, ms in self.updates.items():
            aggr = ms.aggregator
            w = 1.0 / np.sqrt(1.0 + (cur - v)) if self.staleness_weight else 1.0
            if aggr.count > 0:
                for k, val in aggr.dict.items():
                    acc[k] = acc.get(k, 0.0) + w * np.asarray(val)
            num += aggr.count
            ms.aggregator.reset(fl_ctx)
        self.current_model_version += 1
        g = self.current_model.data
        new_model = {k: np.asarray(g[k]) + (acc[k] / num) * self.global_lr if (num > 0 and k in acc) else np.asarray(g[k])
                     for k in g}
        self.updates[self.current_model_version] = _ModelState(ModelUpdateDXOAggregator())
        self.log_info(fl_ctx, f"[paper] generated new model version {self.current_model_version} with {num} updates")
        self.current_model = DXO(data_kind=DataKind.WEIGHTS, data=new_model)
        self.num_updates_counter = 0
        fl_ctx.set_prop(AppConstants.GLOBAL_MODEL, make_model_learnable(new_model, {}), private=True, sticky=True)
        fl_ctx.set_prop(AppConstants.CURRENT_ROUND, self.current_model_version, private=True, sticky=True)
        self.fire_event(AppEventType.GLOBAL_WEIGHTS_UPDATED, fl_ctx)


def make_manager(variant: str, K: int, global_lr: float, staleness_weight: bool):
    cls = {"official": BuffModelManager, "paper": PaperFedBuffModelManager}[variant]
    return cls(num_updates_for_model=K, global_lr=global_lr, staleness_weight=staleness_weight)


def submit_update(mgr, diff: dict, version: int, device: int, vtime: float, fl_ctx) -> bool:
    """Wrap a device model difference as the official ModelUpdate and process it.
    Returns True if a new global model version was generated."""
    before = mgr.current_model_version
    shareable = DXO(data_kind=DataKind.WEIGHT_DIFF, data={"dict": diff, "count": 1}).to_shareable()
    mgr.process_updates({version: ModelUpdate(model_version=version, update=shareable,
                                              devices={f"device-{device}": vtime})}, fl_ctx)
    return mgr.current_model_version != before


def current_weights(mgr) -> dict:
    return {k: np.asarray(v, dtype=np.float32) for k, v in mgr.current_model.data.items()}
