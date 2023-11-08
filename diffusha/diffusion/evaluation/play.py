import json
from typing import Callable

from tqdm import tqdm

from diffusha.actor import LunarLanderKeyboardActor
from diffusha.actor.assistive import DiffusionAssistedActor
from diffusha.actor.base import Actor
from diffusha.config.default_args import Args
from diffusha.diffusion.ddpm import DiffusionModel

# from diffusha.diffusion.evaluation.eval import evaluate
from diffusha.diffusion.evaluation.helper import prepare_diffusha
from diffusha.utils import patch


def eval_assisted_actors(
    diffusion: DiffusionModel,
    make_env: Callable,
    fwd_diff_ratio: float,
    num_episodes: int = 10,
):
    """
    Create DiffusionAssistedActor for each expert obtained from get_actors func,
    and evaluate them with evaluate function.
    """

    sample_env = make_env()
    if hasattr(sample_env, "copilot_observation_space"):
        obs_space = sample_env.copilot_observation_space
    else:
        obs_space = sample_env.observation_space
    act_space = sample_env.action_space

    user_actor = LunarLanderKeyboardActor(sample_env)
    actor = DiffusionAssistedActor(
        obs_space, act_space, diffusion, user_actor, fwd_diff_ratio
    )

    # env = make_env(test=True, seed=0, time_limit=TIME_LIMIT)
    sample_env = make_env(seed=0)

    for _ in tqdm(range(num_episodes)):
        done = False
        obs = sample_env.reset()
        while not done:
            action, diff = actor.act(obs, report_diff=True)
            obs, rew, done, info = sample_env.step(action)
            sample_env.render()


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    from diffusha.data_collection.env import make_env
    from diffusha.utils.reproducibility import set_deterministic

    set_deterministic()  # NOTE: moved from toplevel to here

    parser = argparse.ArgumentParser()

    parser.add_argument("--sweep", help="sweep file")
    parser.add_argument(
        "-l", "--line-number", type=int, help="line number of the sweep-file"
    )
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
    parser.add_argument("--out-dir", help="Output directory")
    parser.add_argument(
        "--save-video", action="store_true", help="Save videos of the episodes"
    )
    parser.add_argument(
        "--force", action="store_true", help="force to overwrite the existing file"
    )
    args = parser.parse_args()
    args.num_episodes = 20

    fwd_diff_ratio = Args.fwd_diff_ratio

    # timestamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S.%f")
    directory = Path(Args.results_dir) / "assistance" / args.env_name.lower()
    directory.mkdir(mode=0o775, parents=True, exist_ok=True)

    # Read config (Args) from config.json
    with open(Path(__file__).parent / "configs.json", "r") as f:
        env2config = json.load(f)

    config = env2config[args.env_name]
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

    saved_keys = [
        "fwd_diff_ratio",
        "obs_noise_level",
        "obs_noise_cfg_prob",
    ]
    config = {}
    config["args"] = {
        key: val
        for key, val in vars(args).items()
        if (
            key not in ["sweep_file", "line_number", "force", "dir_name", "num_env"]
            and not key.startswith("_")
        )
    }
    config["Args"] = {
        key: val
        for key, val in vars(Args).items()
        if (key in saved_keys and not key.startswith("_"))
    }

    env_name = args.env_name
    if "maze2d" in env_name and "goal" in env_name:
        # Fix the goal to bottom left if it is maze2d env
        def make_eval_env(**kwargs):
            return make_env(
                env_name,
                test=True,
                terminate_at_any_goal=True,
                bigger_goal=True,
                goal="left",
                **kwargs,
            )

    elif "LunarLander" in env_name:

        def make_eval_env(**kwargs):
            return make_env(
                env_name,
                test=True,
                split_obs=True,
                **kwargs,
            )

    elif "Push" in env_name:

        def make_eval_env(**kwargs):
            return make_env(
                env_name,
                test=True,
                user_goal="target",
                **kwargs,
            )

    else:
        raise RuntimeError()

    eval_assisted_actors(
        diffusion,
        make_eval_env,
        fwd_diff_ratio=fwd_diff_ratio,
        num_episodes=args.num_episodes,
    )
