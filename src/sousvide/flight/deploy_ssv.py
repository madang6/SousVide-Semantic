import os
import json
import yaml
import pickle
from typing import List, Union, Literal
from re import T, X

import numpy as np
import torch
from torchvision.io import write_video
from torchvision.transforms import Resize
from acados_template import AcadosSimSolver

from controller.pilot import Pilot
from controller.vr_mpc import VehicleRateMPC

import dynamics.quadcopter_config as qc

from synthesize.solvers import min_snap as ms
import synthesize.nerf_utils as nf
import synthesize.trajectory_helper as th
import synthesize.generate_data as gd
from synthesize.build_rrt_dataset import get_objectives, generate_rrt_paths

import visualize.plot_synthesize as ps
import visualize.record_flight as rf

import flight.vision_preprocess as vp


def simulate_roster(cohort: str,
                    scene_name: str,
                    drone_name: str,
                    method_name: str,
                    pilot_list: List[str],
                    nerf_model: nf.NeRF,
                    n_iter_rrt: int = 1000,
                    n_samples: int = 10,
                    save_flight_record: bool = False,
                    review_mode: bool = True):
    """
    Simulate a list of pilots on a given scene, drone, and planning method.
    """

    # === 1) Header ===
    print("=" * 70)
    print(f" Scene : {scene_name}")
    print(f" Method: {method_name}")

    # === 2) Paths & Output Directory ===
    ws = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    courses_cfg_dir = os.path.join(ws, "configs", "courses")
    scenes_cfg_dir  = os.path.join(ws, "configs", "scenes")
    drones_cfg_dir  = os.path.join(ws, "configs", "drones")
    methods_cfg_dir = os.path.join(ws, "configs", "methods")
    output_dir      = os.path.join(ws, "cohorts", cohort, "output")
    os.makedirs(output_dir, exist_ok=True)

    # === 3) Load Drone & Method Configs ===
    with open(os.path.join(drones_cfg_dir, f"{drone_name}.json")) as f:
        drone_cfg = json.load(f)

    with open(os.path.join(methods_cfg_dir, f"{method_name}.json")) as f:
        method_cfg = json.load(f)

    sample_cfg     = method_cfg["sample_set"]
    trajectory_cfg = method_cfg["trajectory_set"]
    drone_set_cfg  = method_cfg["drone_set"]
    if not sample_cfg.get("rrt_mode", False):
        raise NotImplementedError("method config must specify rrt_mode = true")

    # === 4) Load Scene YAML ===
    scene_cfg_file = os.path.join(scenes_cfg_dir, f"{scene_name}.yml")
    with open(scene_cfg_file) as f:
        scene_cfg = yaml.safe_load(f)

    objectives      = scene_cfg["queries"]
    radii           = [(scene_cfg["r1"], scene_cfg["r2"])]
    hover_mode      = scene_cfg["hoverMode"]
    visualize_scene = scene_cfg["visualize"]
    altitude        = scene_cfg["altitude"]
    env_bounds      = {}
    if "minbound" in scene_cfg and "maxbound" in scene_cfg:
        env_bounds["min"] = np.array(scene_cfg["minbound"])
        env_bounds["max"] = np.array(scene_cfg["maxbound"])

    # === 5) Unpack Noise & Simulation Parameters ===
    mu_model   = np.array(sample_cfg["model_noise"]["mean"])
    std_model  = np.array(sample_cfg["model_noise"]["std"])
    mu_sensor  = np.array(sample_cfg["sensor_noise"]["mean"])
    std_sensor = np.array(sample_cfg["sensor_noise"]["std"])
    outdoors   = sample_cfg["outdoors"]
    use_clip   = sample_cfg["clipseg"]
    hz_control = sample_cfg["simulation"]["hz_ctl"]
    hz_sim     = sample_cfg["simulation"]["hz_sim"]
    delay_sim  = sample_cfg["simulation"]["delay"]

    # === 6) Vision Processor ===
    if use_clip:
        vision_processor = vp.CLIPSegONNXModel(
            onnx_path=os.path.join(ws, "cohorts", "clipseg.onnx"),
            hf_model="CIDAS/clipseg-rd64-refined"
        )
    else:
        vision_processor = None

    # === 7) Build or Load Trajectories ===
    combined_prefix = os.path.join(courses_cfg_dir, scene_name)

    if not review_mode:
        # 7a) Generate RRT objectives
        obj_targets, epcds_list, epcds_arr = get_objectives(
            nerf_model, objectives, visualize_scene
        )

        # 7b) Compute goal poses & centroids
        goal_poses, obj_centroid = th.process_RRT_objectives(
            obj_targets, epcds_arr, env_bounds, radii, altitude
        )

        # 7c) Generate raw RRT* branches
        raw_paths = generate_rrt_paths(
            scene_cfg_file, epcds_list, epcds_arr, objectives,
            goal_poses, obj_centroid, env_bounds, n_iter_rrt
        )

        # 7d) Filter, parameterize, and save one trajectory per object
        for obj_name, branches in raw_paths.items():
            alt_set  = th.set_RRT_altitude(branches, altitude)
            filtered = th.filter_branches(alt_set, hover_mode)
            print(f"{obj_name}: {len(filtered)} branches")

            idx = np.random.randint(len(filtered))
            print(f"Selected branch index for {obj_name}: {idx}")

            traj_list, node_list, debug_info = th.parameterize_RRT_trajectories(
                filtered, obj_centroid[idx], 1.0, 20, idx
            )
            print(f"Parameterized: {len(traj_list)} trajectories")

            chosen_traj  = traj_list[idx]
            chosen_nodes = node_list[idx]
            combined_data = {
                "tXUi": chosen_traj,
                "nodes": chosen_nodes,
                **debug_info
            }

            combined_file = f"{combined_prefix}_{obj_name}.pkl"
            with open(combined_file, "wb") as f:
                pickle.dump(combined_data, f)

        # assume single-objective, load last
        combined_file_path = combined_file
        trajectory_data    = combined_data

    else:
        combined_file_path = f"{combined_prefix}_{objectives[0]}.pkl"
        with open(combined_file_path, "rb") as f:
            trajectory_data = pickle.load(f)

        ps.debug_figures_RRT(
            trajectory_data["obj_loc"],
            trajectory_data["positions"],
            trajectory_data["trajectory"],
            trajectory_data["smooth_trajectory"],
            trajectory_data["times"]
        )

    # === 8) Extract Reference Trajectory ===
    tXUi   = trajectory_data["tXUi"]
    t0, tf = tXUi[0, 0], tXUi[0, -1]
    x0     = tXUi[1:11, 0]

    # === 9) Initialize MPC Expert & Transform ===
    base_cfg    = qc.generate_preset_config(drone_cfg)
    mpc_expert  = VehicleRateMPC(tXUi, base_cfg, hz_control)
    transform   = Resize((720, 1280), antialias=True)
    mpc_expert.clear_generated_code()

    # prepend expert to pilots
    pilot_list = ["expert"] + pilot_list

    # === 10) Generate Drone Instances ===
    drone_instances = gd.generate_drones(n_samples, drone_set_cfg, base_cfg)

    # === 11) Simulation Loop ===
    for pilot in pilot_list:
        print("-" * 70)
        print(f"Simulating pilot: {pilot}")

        traj_file = os.path.join(output_dir, f"sim_data_{scene_name}_{pilot}.pt")
        vid_file  = os.path.join(output_dir, f"sim_video_{scene_name}_{pilot}.mp4")

        results = []
        for drone_cfg_inst in drone_instances:
            sim = VehicleRateMPC(tXUi, drone_cfg_inst, hz_control).generate_simulator(hz_sim)

            if pilot == "expert":
                policy = mpc_expert
            else:
                policy = Pilot(cohort, pilot)
                policy.set_mode('deploy')

            Tro, Xro, Uro, images_dict, Tsol, Adv = simulate_flight(
                policy, sim, t0, tf, x0,
                np.zeros((18, 1)),
                nerf_model, objectives[0],
                hz_sim,
                mu_model, std_model,
                mu_sensor, std_sensor,
                delay_sim, outdoors,
                vision_processor
            )

            rollout = {
                "Tro": Tro, "Xro": Xro, "Uro": Uro,
                "Xid": tXUi[1:11, :],
                "obj": np.zeros((18, 1)),
                "Ndata": Uro.shape[1],
                "Tsol": Tsol, "Adv": Adv,
                "rollout_id": "acados_rollout",
                "course": scene_name,
                "drone": drone_name,
                "method": method_name
            }
            results.append(rollout)
            mpc_expert.clear_generated_code()

        semantic_imgs = images_dict["semantic"]

        if save_flight_record:
            fr = rf.FlightRecorder(
                Xro.shape[0], Uro.shape[0],
                20, tXUi[0, -1],
                [224, 398, 3],
                np.zeros((18, 1)),
                cohort, scene_name, pilot
            )
            fr.simulation_import(semantic_imgs, Tro, Xro, Uro, tXUi, Tsol, Adv)
            fr.save()
        else:
            print(f"Simulated {len(drone_instances)} rollouts.")
            torch.save(results, traj_file)

            # prepare and write video
            frames = torch.zeros((semantic_imgs.shape[0], 720, 1280, 3))
            imgs_t = torch.from_numpy(semantic_imgs)
            for i in range(imgs_t.shape[0] - 1):
                img = imgs_t[i].permute(2, 0, 1)
                img = transform(img)
                frames[i] = img.permute(1, 2, 0)

            write_video(vid_file, frames, fps=mpc_expert.hz)
