# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym import LEGGED_GYM_ROOT_DIR
import os

import isaacgym
from isaacgym import gymapi
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger

import numpy as np
import torch

def play(args):
    # Force headless mode
    args.headless = True
    
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 1) # Only 1 env for recording
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()
    
    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)
    
    # --- SETUP HEADLESS CAMERA ---
    print("Setting up Headless Camera...")
    camera_props = gymapi.CameraProperties()
    camera_props.width = 1280
    camera_props.height = 720
    camera_props.enable_tensors = False # Use CPU rendering for saving to disk
    
    # Create camera handle attached to the first environment
    camera_handle = env.gym.create_camera_sensor(env.envs[0], camera_props)
    
    # Position the camera
    # Fixed position relative to origin (easier for start)
    cam_pos = gymapi.Vec3(2.0, 2.0, 1.0)
    cam_target = gymapi.Vec3(0.0, 0.0, 0.5)
    env.gym.set_camera_location(camera_handle, env.envs[0], cam_pos, cam_target)
    
    # Prepare Output Directory
    frame_dir = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'recorded_frames')
    if os.path.exists(frame_dir):
        import shutil
        shutil.rmtree(frame_dir)
    os.makedirs(frame_dir, exist_ok=True)
    print(f"Recording frames to: {frame_dir}")

    # -----------------------------

    logger = Logger(env.dt)
    robot_index = 0 
    joint_index = 1 
    
    # Record for 200 frames (approx 4-5 seconds)
    num_frames = 200
    print(f"Starting recording of {num_frames} frames...")

    for i in range(num_frames):
        actions = policy(obs.detach())
        obs, _, rews, dones, infos = env.step(actions.detach())
        
        # --- RENDER AND SAVE FRAME ---
        # Update graphics
        env.gym.step_graphics(env.sim)
        env.gym.render_all_camera_sensors(env.sim)
        
        # Move camera to track robot (Simple tracking)
        # Get robot position
        robot_pos = env.root_states[0, :3].cpu().numpy()
        cam_pos = gymapi.Vec3(robot_pos[0] + 2.0, robot_pos[1] + 2.0, 1.0)
        cam_target = gymapi.Vec3(robot_pos[0], robot_pos[1], 0.5)
        env.gym.set_camera_location(camera_handle, env.envs[0], cam_pos, cam_target)
        
        # Save image
        filename = os.path.join(frame_dir, f"frame_{i:04d}.png")
        env.gym.write_camera_image_to_file(env.sim, camera_handle, gymapi.IMAGE_COLOR, filename)
        
        if i % 10 == 0:
            print(f"Captured frame {i}/{num_frames}")
        # -----------------------------

    print(f"Done! Frames saved to {frame_dir}")

if __name__ == '__main__':
    args = get_args()
    play(args)
