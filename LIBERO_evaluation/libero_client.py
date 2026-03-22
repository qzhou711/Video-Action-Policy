"""LIBERO simulation client for mimic-video evaluation.

Runs the LIBERO simulation environment and communicates with the mimic-video
model server via WebSocket to get action predictions.

Usage:
    conda activate rynnvla
    python LIBERO_evaluation/libero_client.py \
        --suites libero_spatial libero_object libero_goal libero_10

NOTE: The model server must be running first:
    conda activate mimic
    python scripts/eval_server.py \
        --stage1_checkpoint checkpoints/stage1/final \
        --stage2_checkpoint checkpoints/stage2/final
"""

import os
import sys
import json
import math
import asyncio
import argparse
import logging
import pathlib
import random
import numpy as np

try:
    import websockets
except ImportError:
    print("ERROR: websockets not installed. Run: pip install websockets")
    sys.exit(1)

try:
    import imageio
except ImportError:
    imageio = None

from libero.libero import benchmark
from libero.libero.benchmark import get_libero_path
from libero.libero.envs import OffScreenRenderEnv

# ========= Argument Parsing =========
parser = argparse.ArgumentParser(description="LIBERO evaluation client")
parser.add_argument("--server_url", type=str, default="ws://localhost:8765")
parser.add_argument("--suites", nargs="+",
                    default=["libero_spatial", "libero_object", "libero_goal", "libero_10"])
parser.add_argument("--max_steps", type=int, default=600)
parser.add_argument("--num_episodes", type=int, default=20)
parser.add_argument("--action_horizon", type=int, default=16,
                    help="Number of actions to execute from each predicted chunk")
parser.add_argument("--resolution", type=int, default=256)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--save_video", action="store_true")
parser.add_argument("--video_dir", type=str, default="eval_videos")
parser.add_argument("--log_file", type=str, default="libero_eval.log")
args = parser.parse_args()

# ========= Logging =========
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(args.log_file, mode="a"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)


# ========= Helpers =========
def quat2axisangle(quat):
    """Convert quaternion [x, y, z, w] to axis-angle [ax, ay, az]."""
    w = float(np.clip(quat[3], -1.0, 1.0))
    den = np.sqrt(1.0 - w * w)
    if math.isclose(den, 0.0):
        return np.zeros(3, dtype=np.float64)
    angle = 2.0 * math.acos(w)
    return (quat[:3] * angle) / den


def extract_obs_data(obs):
    """Extract images and state from LIBERO observation dict.

    Returns:
        agentview: [H, W, 3] uint8
        wrist: [H, W, 3] uint8
        state: [8] float64 = [eef_pos(3), axis_angle(3), gripper_qpos(2)]
    """
    # Images: LIBERO renders them upside-down, need flip
    agentview = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])

    # State: match training data format [eef_pos(3), axis_angle(3), gripper_qpos(2)]
    state = np.concatenate([
        obs["robot0_eef_pos"],                       # 3D position
        quat2axisangle(obs["robot0_eef_quat"]),      # quaternion → axis-angle (3D)
        obs["robot0_gripper_qpos"],                   # 2D gripper
    ])

    return agentview, wrist, state


def get_libero_env(task, resolution=256, seed=42):
    """Create LIBERO environment for a specific task."""
    task_bddl_file = (
        pathlib.Path(get_libero_path("bddl_files"))
        / task.problem_folder
        / task.bddl_file
    )
    env = OffScreenRenderEnv(
        bddl_file_name=str(task_bddl_file),
        camera_heights=resolution,
        camera_widths=resolution,
    )
    env.seed(seed)
    return env, task.language


def save_video_file(frames, filename, fps=20, save_dir="eval_videos"):
    if imageio is None or len(frames) == 0:
        return
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, filename)
    imageio.mimsave(filepath, frames, fps=fps)
    log.info(f"Video saved: {filepath} ({len(frames)} frames)")


DUMMY_ACTION = [0.0] * 7


# ========= Main Evaluation Loop =========
async def evaluate_suite(server_url, suite_name, max_steps, num_episodes, action_horizon):
    """Evaluate all tasks in a LIBERO suite."""
    benchmark_dict = benchmark.get_benchmark_dict()
    if suite_name not in benchmark_dict:
        log.error(f"Suite '{suite_name}' not found. Available: {list(benchmark_dict.keys())}")
        return {}

    task_suite = benchmark_dict[suite_name]()
    num_tasks = task_suite.n_tasks
    log.info(f"\n{'='*60}")
    log.info(f"Suite: {suite_name} | Tasks: {num_tasks} | Episodes/task: {num_episodes}")
    log.info(f"{'='*60}")

    suite_results = {}
    total_success = 0
    total_episodes = 0

    async with websockets.connect(
        server_url, max_size=100 * 1024 * 1024, ping_timeout=300,
    ) as ws:
        for task_id in range(num_tasks):
            task = task_suite.get_task(task_id)
            init_states = task_suite.get_task_init_states(task_id)
            env, task_description = get_libero_env(
                task, resolution=args.resolution, seed=args.seed
            )

            log.info(f"\n--- Task {task_id + 1}/{num_tasks}: {task_description} ---")

            task_success = 0
            task_episodes = min(num_episodes, len(init_states))

            for ep in range(task_episodes):
                # Reset environment
                env.reset()
                obs = env.set_init_state(init_states[ep])

                # Warmup steps (let physics stabilize)
                for _ in range(10):
                    obs, _, _, _ = env.step(DUMMY_ACTION)

                # Signal frame buffer reset
                await ws.send(json.dumps({"reset": True}))
                await ws.recv()

                episode_done = False
                success = False
                frames = []
                step_count = 0

                # Collect initial frames to fill the buffer
                # Send multiple initial observations so the model has temporal context
                agentview, wrist, state = extract_obs_data(obs)
                init_frame_msg = {
                    "add_frame": True,
                    "image": [
                        agentview.astype(np.uint8).tolist(),
                        wrist.astype(np.uint8).tolist(),
                    ],
                }
                # Pre-fill buffer with initial frame (repeated)
                for _ in range(16):
                    await ws.send(json.dumps(init_frame_msg))
                    await ws.recv()

                while step_count < max_steps:
                    # === Query model for action chunk ===
                    agentview, wrist, state = extract_obs_data(obs)
                    query_msg = {
                        "query": True,
                        "image": [
                            agentview.astype(np.uint8).tolist(),
                            wrist.astype(np.uint8).tolist(),
                        ],
                        "state": state.tolist(),
                        "prompt": task_description,
                    }
                    await ws.send(json.dumps(query_msg))
                    result = await ws.recv()

                    try:
                        action_chunk = json.loads(result)
                    except (json.JSONDecodeError, ValueError) as e:
                        log.error(f"Action parsing failed: {e}")
                        break

                    # === Execute action chunk ===
                    for i in range(min(action_horizon, len(action_chunk))):
                        action = list(action_chunk[i])

                        # Gripper: training data uses {-1, 1} directly
                        # LIBERO also expects [-1, 1] for gripper
                        # Clip gripper to valid range, threshold to binary
                        if action[6] >= 0:
                            action[6] = 1.0    # open
                        else:
                            action[6] = -1.0   # close

                        try:
                            obs, reward, done, info = env.step(action[:7])
                        except ValueError as ve:
                            log.error(f"Invalid action: {ve}")
                            episode_done = True
                            break

                        step_count += 1

                        # Send intermediate frame to server to maintain temporal context
                        agentview_mid, wrist_mid, _ = extract_obs_data(obs)
                        frame_msg = {
                            "add_frame": True,
                            "image": [
                                agentview_mid.astype(np.uint8).tolist(),
                                wrist_mid.astype(np.uint8).tolist(),
                            ],
                        }
                        await ws.send(json.dumps(frame_msg))
                        await ws.recv()

                        # Record video
                        if args.save_video:
                            frame = np.hstack([
                                np.rot90(obs["agentview_image"], 2),
                                np.rot90(obs["robot0_eye_in_hand_image"], 2),
                            ])
                            frames.append(frame)

                        if done:
                            episode_done = True
                            success = True
                            task_success += 1
                            total_success += 1
                            break

                    if episode_done or step_count >= max_steps:
                        break

                # Save video
                if args.save_video:
                    save_video_file(
                        frames,
                        f"task{task_id + 1}_ep{ep + 1}.mp4",
                        fps=30,
                        save_dir=os.path.join(args.video_dir, suite_name),
                    )

                status = "✅ Success" if success else "❌ Fail"
                log.info(f"  Episode {ep + 1}/{task_episodes}: {status} (steps={step_count})")

            task_sr = task_success / task_episodes if task_episodes > 0 else 0
            suite_results[task_description] = {
                "success": task_success,
                "total": task_episodes,
                "success_rate": task_sr,
            }
            total_episodes += task_episodes
            log.info(f"  Task {task_id + 1} SR: {task_success}/{task_episodes} = {task_sr:.1%}")
            env.close()

    suite_sr = total_success / total_episodes if total_episodes > 0 else 0
    log.info(f"\n{'='*60}")
    log.info(f"Suite {suite_name} Summary: {total_success}/{total_episodes} = {suite_sr:.1%}")
    log.info(f"{'='*60}")

    suite_results["__summary__"] = {
        "suite": suite_name,
        "total_success": total_success,
        "total_episodes": total_episodes,
        "average_success_rate": suite_sr,
    }
    return suite_results


async def main():
    np.random.seed(args.seed)
    random.seed(args.seed)

    all_results = {}
    for suite_name in args.suites:
        log.info(f"\n{'#'*60}")
        log.info(f"# Evaluating suite: {suite_name}")
        log.info(f"{'#'*60}")

        results = await evaluate_suite(
            server_url=args.server_url,
            suite_name=suite_name,
            max_steps=args.max_steps,
            num_episodes=args.num_episodes,
            action_horizon=args.action_horizon,
        )
        all_results[suite_name] = results

    # Final summary
    log.info(f"\n{'#'*60}")
    log.info("# FINAL RESULTS")
    log.info(f"{'#'*60}")
    for suite_name, results in all_results.items():
        summary = results.get("__summary__", {})
        sr = summary.get("average_success_rate", 0)
        total = summary.get("total_episodes", 0)
        success = summary.get("total_success", 0)
        log.info(f"  {suite_name:20s}: {success}/{total} = {sr:.1%}")

    with open("libero_eval_results.json", "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    log.info("\nResults saved to libero_eval_results.json")


if __name__ == "__main__":
    asyncio.run(main())
