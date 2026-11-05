"""Step-1 sanity: run the official FedAvgRecipe in SimEnv with real training."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
os.environ["FEDASMU_ROOT"] = ROOT

from nvflare.app_opt.pt.recipes.fedavg import FedAvgRecipe  # noqa: E402
from nvflare.recipe.sim_env import SimEnv  # noqa: E402

from src.models import build_model  # noqa: E402

if __name__ == "__main__":
    recipe = FedAvgRecipe(name="sanity_fedavg", model=build_model("cnn", 10), min_clients=2, num_rounds=3,
                          train_script=os.path.join(os.path.dirname(os.path.abspath(__file__)), "client.py"))
    env = SimEnv(num_clients=2, workspace_root=os.path.join(ROOT, "workspaces"))
    run = recipe.execute(env)
    print("result:", run.get_result())
