"""
Records a GIF of a policy with MuJoCo's offscreen renderer (no display needed: OSMesa, software rendering).

    python3 -m scripts.make_gif --model pretrained/best.pth --out results/gifs/stand.gif
    python3 -m scripts.make_gif --model <running ckpt> --speed 0.6 --assist 0.3 --out results/gifs/run.gif

Needs the system library libOSMesa (apt-get install libosmesa6). The camera follows the torso from the side;
the overlay shows the time, the speed against the command (running task), the assist and the pushes.
"""
import argparse
import os
import sys

# torch must be fully imported before MuJoCo: once MuJoCo has loaded OSMesa, importing triton in the same process
# segfaults (both libraries bundle their own LLVM). torch.optim.Adam imports it lazily, hence torch._dynamo here.
import numpy as np
import torch
import torch._dynamo  # noqa: F401

os.environ.setdefault("MUJOCO_GL", "osmesa")  # the backend is chosen when MuJoCo is first imported

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import mujoco
from PIL import Image, ImageDraw, ImageFont

from agents.loading import load_agent, task_of
from envs.registry import env_class


def make_camera(env, distance: float, azimuth: float, elevation: float) -> mujoco.MjvCamera:
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = env._torso_id
    cam.distance, cam.azimuth, cam.elevation = distance, azimuth, elevation
    return cam


def caption(env, task: str, t: float) -> str:
    push = float(np.linalg.norm(env.data.xfrc_applied[env._torso_id, :2]))
    text = f"t = {t:4.1f} s   height {env.data.qpos[2]:.2f} m"
    if task == "run":
        com = env.data.subtree_linvel[env._torso_id]
        d = np.array([np.cos(env._cmd_yaw), np.sin(env._cmd_yaw)])
        text = (f"t = {t:4.1f} s   speed {com[:2] @ d:4.2f} / {env._v_target:4.2f} m/s   "
                f"assist {env._assist:.2f}")
    if push > 0:
        text += f"   PUSH {push:3.0f} N"
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description="Record a GIF of a policy.")
    parser.add_argument("--model", type=str, required=True, help="PPO checkpoint (either task)")
    parser.add_argument("--out", type=str, default="results/gifs/policy.gif")
    parser.add_argument("--seconds", type=float, default=6.0, help="Simulated seconds")
    parser.add_argument("--every", type=int, default=2, help="Render one frame every N control steps (2 = real time at 33 fps)")
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--perturbation", type=float, default=0.0,
                        help="Intensity in [0, 1] of the reset randomization, domain randomization and pushes")
    parser.add_argument("--speed", type=float, default=0.6, help="[run] commanded speed (m/s)")
    parser.add_argument("--heading", type=float, default=0.0, help="[run] heading offset of the command (rad)")
    parser.add_argument("--assist", type=float, default=0.0, help="[run] strength of the assist in [0, 1]")
    parser.add_argument("--stochastic", action="store_true", help="Sample the actions (default: use the mean)")
    parser.add_argument("--distance", type=float, default=4.0, help="Camera distance (m)")
    parser.add_argument("--azimuth", type=float, default=90.0, help="Camera azimuth (deg): 90 = side view")
    parser.add_argument("--elevation", type=float, default=-10.0)
    parser.add_argument("--seed", type=int, default=3)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    agent, ckpt, env_kwargs = load_agent(args.model)
    task = task_of(ckpt)
    env = env_class(task)(render=False, seed=args.seed, **env_kwargs)
    if task == "run":
        env.set_difficulty(1.0)                       # the full task; the assist and perturbations are set explicitly
        env.set_assist(args.assist)
        env.set_command(args.speed, args.heading)
        env.set_perturbation(args.perturbation, random_amplitude=False)
    elif args.perturbation > 0:
        env.set_difficulty(args.perturbation, random_amplitude=False)

    renderer = mujoco.Renderer(env.model, args.height, args.width)
    camera = make_camera(env, args.distance, args.azimuth, args.elevation)
    font = ImageFont.load_default(size=max(10, args.height // 18))

    def frame(t: float) -> Image.Image:
        renderer.update_scene(env.data, camera=camera)
        image = Image.fromarray(renderer.render().copy())
        draw = ImageDraw.Draw(image)
        draw.rectangle([0, 0, args.width, font.size + 8], fill=(0, 0, 0))
        draw.text((6, 3), caption(env, task, t), fill=(255, 255, 255), font=font)
        return image

    obs = env.reset()
    frames = [frame(0.0)]
    fell_at = None
    for step in range(1, int(args.seconds / env.dt) + 1):
        action, _ = agent.select_action(obs, training=args.stochastic)
        obs, _, done, _ = env.step(action)
        if step % args.every == 0:
            frames.append(frame(step * env.dt))
        if done:
            fell_at = step * env.dt
            break
    if fell_at is not None:  # keep the last frame on screen for a moment
        frames.extend([frame(fell_at)] * max(1, int(1.0 / (env.dt * args.every))))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    duration = int(round(1000 * env.dt * args.every))
    frames[0].save(args.out, save_all=True, append_images=frames[1:], duration=duration, loop=0, optimize=True)
    size = os.path.getsize(args.out) / 1e6
    print(f"{args.out}: {len(frames)} frames, {size:.1f} MB, task {task}, "
          + (f"fell after {fell_at:.1f} s" if fell_at is not None else f"stayed up for {args.seconds:.1f} s"))


if __name__ == "__main__":
    main()
