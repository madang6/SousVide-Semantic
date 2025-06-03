import os
os.environ["ACADOS_SOURCE_DIR"] = "/data/madang/StanfordMSL/SousVide-Semantic/FiGS-Semantic/acados"
os.environ["LD_LIBRARY_PATH"] = os.getenv("LD_LIBRARY_PATH", "") + "/data/madang/StanfordMSL/SousVide-Semantic/FiGS-Semantic/acados/lib"
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"  # Disable Albumentations update check

import typer
import yaml
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable, List, Optional
import wandb

from io import BytesIO
from PIL import Image

import matplotlib.pyplot as plt
import plotly.graph_objects as go

import sousvide.synthesize.rollout_generator as rg
import sousvide.synthesize.observation_generator as og
import sousvide.instruct.train_policy as tp
import sousvide.visualize.plot_synthesize as ps
import sousvide.visualize.plot_learning as pl
import sousvide.flight.deploy_ssv as df


app = typer.Typer()

# Monkey-patch Plotly.show to capture figures
_all_plotly_figs: List[go.Figure] = []
_original_show = go.Figure.show

def _capture_and_show(self, *args, **kwargs):
    _all_plotly_figs.append(self)
    return _original_show(self, *args, **kwargs)
go.Figure.show = _capture_and_show


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def init_wandb(cfg: dict, job: str) -> None:
    if cfg.get("use_wandb"):
        wandb.init(
            project=cfg.get("wandb_project", "default_project"),
            name=cfg.get("wandb_run_name", job),
            config=cfg,
        )


def common_options(
    config_file: Path,
    plot: bool,
    use_wandb: bool,
    wandb_project: Optional[str],
    wandb_run_name: Optional[str],
) -> dict:
    cfg = load_yaml(config_file)
    cfg.update({
        "plot": plot,
        "use_wandb": use_wandb,
        "wandb_project": wandb_project,
        "wandb_run_name": wandb_run_name,
    })
    return cfg

@app.command("generate-rollouts")
def generate_rollouts(
    config_file: Path = typer.Option(..., exists=True),
    plot: bool = typer.Option(False),
    use_wandb: bool = typer.Option(False),
    wandb_project: Optional[str] = typer.Option(None),
    wandb_run_name: Optional[str] = typer.Option(None),
):
    cfg = common_options(  # type: ignore
        config_file, plot, use_wandb, wandb_project, wandb_run_name
    )
    init_wandb(cfg, "generate_rollouts")
    rg.generate_rollout_data(cfg["cohort"], cfg["method"], cfg["flights"])
    if cfg["plot"]:
        fig = ps.plot_rollout_data(cfg["cohort"])
        if cfg["use_wandb"]:
            wandb.log({"rollout_plot": fig})

@app.command("generate-observations")
def generate_observations(
    config_file: Path = typer.Option(..., exists=True),
    plot: bool = typer.Option(False),
    use_wandb: bool = typer.Option(False),
    wandb_project: Optional[str] = typer.Option(None),
    wandb_run_name: Optional[str] = typer.Option(None),
):
    cfg = common_options(  # type: ignore
        config_file, plot, use_wandb, wandb_project, wandb_run_name
    )
    init_wandb(cfg, "generate_observations")
    og.generate_observation_data(cfg["cohort"], cfg["roster"])
    if cfg["plot"]:
        fig = ps.plot_observation_data(cfg["cohort"], cfg["roster"])
        if cfg["use_wandb"]:
            wandb.log({"observation_plot": fig})

@app.command("train-history")
def train_history(
    config_file: Path = typer.Option(..., exists=True),
    plot: bool = typer.Option(False),
    use_wandb: bool = typer.Option(False),
    wandb_project: Optional[str] = typer.Option(None),
    wandb_run_name: Optional[str] = typer.Option(None),
):
    cfg = common_options(  # type: ignore
        config_file, plot, use_wandb, wandb_project, wandb_run_name
    )
    init_wandb(cfg, "train_history")
    tp.train_roster(
        cfg["cohort"], cfg["roster"], "Parameter", cfg["Nep_his"], lim_sv=cfg.get("lim_sv", 10)
    )
    if cfg["plot"]:
        fig = pl.plot_losses(cfg["cohort"], cfg["roster"], "Parameter")
        if cfg["use_wandb"]:
            wandb.log({"history_loss_plot": fig})

@app.command("train-command")
def train_command(
    config_file: Path = typer.Option(..., exists=True),
    plot: bool = typer.Option(False),
    use_wandb: bool = typer.Option(False),
    wandb_project: Optional[str] = typer.Option(None),
    wandb_run_name: Optional[str] = typer.Option(None),
):
    cfg = common_options(  # type: ignore
        config_file, plot, use_wandb, wandb_project, wandb_run_name
    )
    init_wandb(cfg, "train_command")
    tp.train_roster(
        cfg["cohort"], cfg["roster"], "Commander", cfg["Nep_com"], lim_sv=cfg.get("lim_sv", 10)
    )
    if cfg["plot"]:
        fig = pl.plot_losses(cfg["cohort"], cfg["roster"], "Commander")
        if cfg["use_wandb"]:
            wandb.log({"command_loss_plot": fig})

@app.command()
def simulate(
    config_file: Path = typer.Option(..., exists=True),
    use_wandb: bool = typer.Option(False),
    wandb_project: Optional[str] = typer.Option(None),
    wandb_run_name: Optional[str] = typer.Option(None),
):
    cfg = common_options(
        config_file, False, use_wandb, wandb_project, wandb_run_name
    )
    init_wandb(cfg, "simulate")
    df.simulate_roster(
        cfg["cohort"], cfg["method"], cfg["flights"], cfg["roster"]
    )

    if cfg.get("use_wandb"):
        logs = {}
        for i, num in enumerate(plt.get_fignums(), start=1):
            fig_mpl = plt.figure(num)
            logs[f"simulate_mpl_fig_{i}"] = wandb.Image(fig_mpl)

        for i, fig in enumerate(_all_plotly_figs, start=1):
            img_bytes = fig.to_image(format="png", width=1200, height=1200)
            buf = BytesIO(img_bytes)
            pil_img = Image.open(buf)
            logs[f"simulate_plotly_png_{i}"] = wandb.Image(pil_img)

        wandb.log(logs)
        plt.close("all")
        _all_plotly_figs.clear()

@app.command()
def debug_trajectory(
    config_file: Path = typer.Option(..., exists=True),
    use_wandb: bool = typer.Option(False),
    wandb_project: Optional[str] = typer.Option(None),
    wandb_run_name: Optional[str] = typer.Option(None),
):
    import os
    import glob
    import pickle
    import numpy as np
    import torch
    import figs.utilities.trajectory_helper as th
    import figs.visualize.plot_trajectories as pt
    # Load and resolve config
    cfg = common_options(config_file, False, use_wandb, wandb_project, wandb_run_name)
    init_wandb(cfg, "debug_trajectory")

    # Determine where scene configs live
    workspace_path = Path(__file__).resolve().parents[1]
    scenes_cfg_dir = workspace_path / "configs" / "scenes"
    cohort_path_base = workspace_path / "cohorts" / cfg["cohort"]

    # For each flight (scene, course)
    for scene_name, _ in cfg["flights"]:
        scene_cfg_file = scenes_cfg_dir / f"{scene_name}.yml"
        with open(scene_cfg_file) as f:
            scene_cfg = yaml.safe_load(f)

        combined_prefix = scenes_cfg_dir / scene_name
        # Find any matching .pkl under the prefix
        for combined_path in glob.glob(f"{combined_prefix}*.pkl"):
            with open(combined_path, "rb") as f:
                data = pickle.load(f)

            # Derive object name from the filename
            base = Path(combined_path).stem
            obj_name = base.replace(f"{scene_name}_", "")

            expert_filename = cohort_path_base / f"sim_data_{scene_name}_{obj_name}_expert.pt"
            if expert_filename.exists():
                expert_data = torch.load(expert_filename)
                typer.echo(f"expert_data length: {len(expert_data)}")
                try:
                    # th.debug_figures_RRT(
                    #     data["obj_loc"],
                    #     data["positions"],
                    #     data["smooth_trajectory"],   
                    #     expert_data[-1]["Xro"],  # Take the last element      
                    #     data["times"],
                    # )
                    pt.plot_RO_time((expert_data[-1]["Tro"], expert_data[-1]["Xro"], expert_data[-1]["Uro"]),
                                    plot_p=False, plot_q=True, aesthetics=False)
                    pt.plot_RO_time((data["tXUi"][0], data["tXUi"][1:11], data["tXUi"][11:15,:-1]),
                                    plot_p=False, plot_q=True, aesthetics=False)
                    # th.debug_figures_RRT(
                    #     data["obj_loc"],
                    #     data["positions"],
                    #     data["trajectory"],
                    #     data["smooth_trajectory"],
                    #     data["times"],
                    # )
                except:
                    typer.echo(f"Error occurred. expert_data[-1] type: {type(expert_data[-1])}")
                    if isinstance(expert_data[-1], dict):
                        typer.echo(f"expert_data[-1] keys: {list(expert_data[-1].keys())}")

                    raise ValueError("Error processing trajectories")
            else:
                # Fallback to the original call if expert file is missing
                th.debug_figures_RRT(
                    data["obj_loc"],
                    data["positions"],
                    data["trajectory"],
                    data["smooth_trajectory"],
                    data["times"],
                )

            def process_quaternions(data_array, label):
                ncols = data_array.shape[1]
                indices = np.linspace(0, ncols - 1, num=10, dtype=int)
                
                for i in indices:
                    qx, qy, qz, qw = data_array[7:11, i]
                    t = data_array[0, i]
                    roll = np.arctan2(2.0 * (qw * qx + qy * qz), 1.0 - 2.0 * (qx * qx + qy * qy))
                    pitch = np.arcsin(np.clip(2.0 * (qw * qy - qz * qx), -1.0, 1.0))
                    yaw = np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
                    typer.echo(f"{obj_name} t={t:.2f}: roll={roll:.4f} rad, pitch={pitch:.4f} rad, yaw={yaw:.4f} rad")

            # Process both tXUi and trajectory data
            process_quaternions(data["tXUi"], "tXUi")
            process_quaternions(data["trajectory"], "trajectory")

    # If using W&B, log any generated figures
    if cfg.get("use_wandb"):
        logs = {}
        # Log any Matplotlib figures
        for i, num in enumerate(plt.get_fignums(), start=1):
            fig_mpl = plt.figure(num)
            logs[f"debug_mpl_fig_{i}"] = wandb.Image(fig_mpl)

        # Log any Plotly figures as PNG via Kaleido
        for i, fig in enumerate(_all_plotly_figs, start=1):
            img_bytes = fig.to_image(format="png", width=1200, height=1200)
            buf = BytesIO(img_bytes)
            pil_img = Image.open(buf)
            logs[f"debug_plotly_png_{i}"] = wandb.Image(pil_img)

        wandb.log(logs)
        plt.close("all")
        _all_plotly_figs.clear()

# @app.command()
# def debug_trajectory(
#     config_file: Path = typer.Option(..., exists=True),
#     use_wandb: bool = typer.Option(False),
#     wandb_project: Optional[str] = typer.Option(None),
#     wandb_run_name: Optional[str] = typer.Option(None),
# ):
#     import glob
#     import os
#     import pickle
#     import numpy as np
#     import torch
#     import figs.utilities.trajectory_helper as th
#     # Load and resolve config
#     cfg = common_options(config_file, False, use_wandb, wandb_project, wandb_run_name)
#     init_wandb(cfg, "debug_trajectory")

#     # Determine where scene configs live
#     workspace_path = Path(__file__).resolve().parents[1]
#     scenes_cfg_dir = workspace_path / "configs" / "scenes"

#     cohort_name = cfg["cohort"]
#     roster = cfg["roster"]

#     # For each flight (scene, course)
#     for scene_name, _ in cfg["flights"]:
#         scene_cfg_file = scenes_cfg_dir / f"{scene_name}.yml"
#         with open(scene_cfg_file) as f:
#             scene_cfg = yaml.safe_load(f)

#         combined_prefix = scenes_cfg_dir / scene_name
#         # Find any matching .pkl under the prefix
#         for combined_path in glob.glob(f"{combined_prefix}*.pkl"):
#             with open(combined_path, "rb") as f:
#                 data = pickle.load(f)

#             # Derive object name from the filename
#             base = Path(combined_path).stem
#             obj_name = base.replace(f"{scene_name}_", "")

#             # If there is a pilot-specific file, load Xro from it
#             for pilot_name in roster:
#                 cohort_path = os.path.join(workspace_path, "cohorts", cohort_name)
#                 traj_file = os.path.join(
#                     cohort_path, f"sim_data_{scene_name}_{obj_name}_{pilot_name}.pt"
#                 )
#                 if os.path.exists(traj_file):
#                     traj_dict = torch.load(traj_file)
#                     Xro = traj_dict["Xro"]
#                 else:
#                     Xro = None

#                 # Plot debug figures for this object, swapping trajectories if Xro exists
#                 if Xro is not None:
#                     th.debug_figures_RRT(
#                         data["obj_loc"],
#                         data["positions"],
#                         data["smooth_trajectory"],  # original smooth → now treat as “trajectory”
#                         Xro,                         # use Xro in place of smooth_trajectory
#                         data["times"]
#                     )
#                 else:
#                     th.debug_figures_RRT(
#                         data["obj_loc"],
#                         data["positions"],
#                         data["trajectory"],
#                         data["smooth_trajectory"],
#                         data["times"]
#                     )

#             def process_quaternions(data_array, label):
#                 ncols = data_array.shape[1]
#                 indices = np.linspace(0, ncols - 1, num=10, dtype=int)
                
#                 for i in indices:
#                     qx, qy, qz, qw = data_array[7:11, i]
#                     t = data_array[0, i]
#                     roll = np.arctan2(2.0 * (qw * qx + qy * qz), 1.0 - 2.0 * (qx * qx + qy * qy))
#                     pitch = np.arcsin(np.clip(2.0 * (qw * qy - qz * qx), -1.0, 1.0))
#                     yaw = np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
#                     typer.echo(f"{obj_name} t={t:.2f}: roll={roll:.4f} rad, pitch={pitch:.4f} rad, yaw={yaw:.4f} rad")

#             # Process both tXUi and trajectory/quaternion data
#             process_quaternions(data["tXUi"], "tXUi")
#             process_quaternions(data["trajectory"], "trajectory")

#     # If using W&B, log any generated figures
#     if cfg.get("use_wandb"):
#         logs = {}
#         # Log any Matplotlib figures
#         for i, num in enumerate(plt.get_fignums(), start=1):
#             fig_mpl = plt.figure(num)
#             logs[f"debug_mpl_fig_{i}"] = wandb.Image(fig_mpl)

#         # Log any Plotly figures as PNG via Kaleido
#         for i, fig in enumerate(_all_plotly_figs, start=1):
#             img_bytes = fig.to_image(format="png", width=1200, height=1200)
#             buf = BytesIO(img_bytes)
#             pil_img = Image.open(buf)
#             logs[f"debug_plotly_png_{i}"] = wandb.Image(pil_img)

#         wandb.log(logs)
#         plt.close("all")
#         _all_plotly_figs.clear()

if __name__ == "__main__":
    app()