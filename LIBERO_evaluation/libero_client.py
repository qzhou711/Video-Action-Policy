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



        python LIBERO_evaluation/libero_client.py \
            --server_url ws://localhost:8765 \
            --suites libero_object \
            --num_episodes 3 \
            --save_video \
            --debug_rollout_log \
            --debug_log_interval 20
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
parser.add_argument("--max_steps", type=int, default=1000)
parser.add_argument("--num_episodes", type=int, default=20)
parser.add_argument("--action_horizon", type=int, default=8, #16
                    help="Number of actions to execute from each predicted chunk")
parser.add_argument("--resolution", type=int, default=256)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--save_video", action="store_true")
parser.add_argument("--video_dir", type=str, default="eval_videos")
parser.add_argument("--log_file", type=str, default="libero_eval.log")
parser.add_argument("--warmup_frames", type=int, default=10,
                    help="Number of real warmup frames to push before first query")
parser.add_argument("--debug_rollout_log", action="store_true",
                    help="Enable verbose rollout logs (reward/done/info snapshots)")
parser.add_argument("--debug_log_interval", type=int, default=50,
                    help="Step interval for rollout debug logs (when --debug_rollout_log is set)")
args = parser.parse_args()
if args.warmup_frames < 1:
    raise ValueError("--warmup_frames must be >= 1")
if args.debug_log_interval < 1:
    raise ValueError("--debug_log_interval must be >= 1")

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
                episode_done = False
                success = False
                frames = []
                step_count = 0
                episode_end_reason = "unknown"

                # Signal frame buffer reset before warmup
                await ws.send(json.dumps({"reset": True}))
                await ws.recv()

                # Warmup steps: let physics stabilize AND collect real frames.
                # Keep this aligned with server-side inference frame window
                # (--num_infer_real_frames, default 5).
                for _ in range(args.warmup_frames):
                    try:
                        obs, _, done, _ = env.step(DUMMY_ACTION)
                    except ValueError as ve:
                        err = str(ve).lower()
                        if "terminated episode" in err:
                            log.warning(
                                "Episode terminated during warmup; skipping rollout "
                                f"(task={task_id + 1}, ep={ep + 1})."
                            )
                            episode_end_reason = "terminated_during_warmup"
                        else:
                            log.error(f"Invalid action during warmup: {ve}")
                            episode_end_reason = "invalid_action_during_warmup"
                        episode_done = True
                        break
                    agentview_w, wrist_w, _ = extract_obs_data(obs)
                    await ws.send(json.dumps({
                        "add_frame": True,
                        "image": [
                            agentview_w.astype(np.uint8).tolist(),
                            wrist_w.astype(np.uint8).tolist(),
                        ],
                    }))
                    await ws.recv()
                    if done:
                        log.warning(
                            "Episode reached done during warmup; skipping rollout "
                            f"(task={task_id + 1}, ep={ep + 1})."
                        )
                        episode_end_reason = "done_during_warmup"
                        episode_done = True
                        break

                gripper_state = -1.0  # start open; sticky gripper state persists across chunks

                while step_count < max_steps and not episode_done:
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
                    # Sticky gripper: only switch state when prediction clearly
                    # crosses a hysteresis threshold, avoiding oscillation near zero.
                    # Confirmed convention: +1 = close, -1 = open (robosuite / [-1,1] range)
                    CLOSE_THRESH =  0.3
                    OPEN_THRESH  = -0.3
                    for i in range(min(action_horizon, len(action_chunk))):
                        action = list(action_chunk[i])

                        g_raw = float(action[6])
                        if g_raw > CLOSE_THRESH:
                            gripper_state = 1.0
                        elif g_raw < OPEN_THRESH:
                            gripper_state = -1.0
                        # within [-0.3, 0.3]: keep previous gripper_state
                        action[6] = gripper_state
                        g_applied = float(action[6])

                        try:
                            obs, reward, done, info = env.step(action[:7])
                        except ValueError as ve:
                            err = str(ve).lower()
                            if "terminated episode" in err:
                                log.warning(
                                    "Episode already terminated before step; "
                                    f"stop current episode (task={task_id + 1}, ep={ep + 1}, step={step_count})."
                                )
                                episode_end_reason = "terminated_before_step"
                            else:
                                log.error(f"Invalid action: {ve}")
                                episode_end_reason = "invalid_action"
                            episode_done = True
                            break

                        step_count += 1
                        if args.debug_rollout_log and (
                            step_count == 1 or step_count % args.debug_log_interval == 0 or done
                        ):
                            log.info(
                                "Rollout debug | task=%d ep=%d step=%d reward=%.4f done=%s g_raw=%.4f g_applied=%.1f info=%s",
                                task_id + 1,
                                ep + 1,
                                step_count,
                                float(reward),
                                str(done),
                                g_raw,
                                g_applied,
                                str(info),
                            )

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
                            episode_end_reason = "success_predicate_triggered"
                            task_success += 1
                            total_success += 1
                            break

                    if episode_done or step_count >= max_steps:
                        if step_count >= max_steps and not success and episode_end_reason == "unknown":
                            episode_end_reason = "max_steps_reached"
                        break

                # Save video
                if args.save_video:
                    outcome_tag = "success" if success else "fail"
                    save_video_file(
                        frames,
                        f"task{task_id + 1}_ep{ep + 1}_{outcome_tag}.mp4",
                        fps=30,
                        save_dir=os.path.join(args.video_dir, suite_name),
                    )

                status = "✅ Success" if success else "❌ Fail"
                if episode_end_reason == "unknown":
                    episode_end_reason = "loop_exited_without_explicit_reason"
                log.info(
                    f"  Episode {ep + 1}/{task_episodes}: {status} "
                    f"(steps={step_count}, reason={episode_end_reason})"
                )

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
