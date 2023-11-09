import argparse
import json
import time
from pathlib import Path

from diffusha.actor import LunarLanderKeyboardActor
from diffusha.actor.assistive import DiffusionAssistedActor
from diffusha.config.default_args import Args
from diffusha.data_collection.env import make_env
from diffusha.diffusion.evaluation.helper import prepare_diffusha
from diffusha.utils.reproducibility import set_deterministic

set_deterministic()  # NOTE: moved from toplevel to here

parser = argparse.ArgumentParser()
parser.add_argument(
    "--env-name",
    default="LunarLander-v1",
    choices=[
        "Maze2d-simple-two-goals-v0",
        "LunarLander-v1",
        "LunarLander-v5",
        "BlockPushMultimodal-v1",
    ],
    help="what env to use",
)
parser.add_argument(
    "--no-assist", action="store_true", help="whether to use assistive actor"
)
parser.add_argument("--fps", type=int, default=30, help="frames per second")
args = parser.parse_args()
args.num_episodes = 20

fwd_diff_ratio = Args.fwd_diff_ratio

# create env
env_name = args.env_name
if "maze2d" in env_name and "goal" in env_name:
    # Fix the goal to bottom left if it is maze2d env
    env_args = {
        "env_name": env_name,
        "test": True,
        "terminate_at_any_goal": True,
        "bigger_goal": True,
        "goal": "left",
    }
elif "LunarLander" in env_name:
    env_args = {
        "env_name": env_name,
        "test": True,
        "split_obs": True,
    }
elif "Push" in env_name:
    env_args = {
        "env_name": env_name,
        "test": True,
        "user_goal": "target",
    }
else:
    raise RuntimeError()

sample_env = make_env(**env_args)

# create actor
actor = LunarLanderKeyboardActor(sample_env)

if not args.no_assist:
    # Read config (Args) from config.json
    with open(Path(__file__).parent / "configs.json", "r") as f:
        env2config = json.load(f)

    env2step = {
        "maze2d-simple-two-goals": 9999,
        "LunarLander-v1": 29999,
        "LunarLander-v5": 24000,
        "BlockPushMultimodal-v1": 29999,
    }

    # Retrieve Args from wandb; NOTE: This updates Args internally!!
    diffusion = prepare_diffusha(
        env2config[args.env_name],
        Path(Args.ddpm_model_path) / args.env_name.lower(),
        env2step[args.env_name],
        args.env_name,
        fwd_diff_ratio,
    )

    if hasattr(sample_env, "copilot_observation_space"):
        obs_space = sample_env.copilot_observation_space
    else:
        obs_space = sample_env.observation_space
    act_space = sample_env.action_space

    actor = DiffusionAssistedActor(
        obs_space, act_space, diffusion, actor, fwd_diff_ratio
    )

# play
env = make_env(**env_args, seed=0)

interval = 1.0 / args.fps

for ep in range(args.num_episodes):
    obs = env.reset()
    done = False
    r_ep = 0.0

    while not done:
        start_time = time.time()

        env.render()
        obs, r, done, _ = env.step(actor.act(obs))
        r_ep += r

        # manage fps
        elapsed = time.time() - start_time
        if elapsed < interval:
            time.sleep(interval - elapsed)

    print(f"episode {ep + 1}: reward = {r_ep:.2f}")
