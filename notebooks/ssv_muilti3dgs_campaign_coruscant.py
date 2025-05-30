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

if __name__ == "__main__":
    app()