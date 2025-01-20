from math import e
from pathlib import Path
import os
from re import T
import torch
import numpy as np
from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.utils.eval_utils import eval_setup
from scipy.spatial.transform import Rotation as R
from typing import Literal, List
import synthesize.trajectory_helper as th

class NeRF():
    def __init__(self, config_path: Path, width:int=640, height:int=360) -> None:
        # config path
        self.config_path = config_path

        # Pytorch Config
        use_cuda = torch.cuda.is_available()                    
        self.device = torch.device("cuda:0" if use_cuda else "cpu")

        # Get config and pipeline
        self.config, self.pipeline, _, _ = eval_setup(
            self.config_path, 
            test_mode="inference",
        )

        # Get reference camera
        self.camera_ref = self.pipeline.datamanager.eval_dataset.cameras[0]

        # Render parameters
        self.channels = 3
        self.camera_out,self.width,self.height = self.generate_output_camera(width,height)

    def generate_output_camera(self,width:int,height:int):
        fx,fy = 462.956,463.002
        cx,cy = 323.076,181.184
        
        camera_out = Cameras(
            camera_to_worlds=1.0*self.camera_ref.camera_to_worlds,
            fx=fx,fy=fy,
            cx=cx,cy=cy,
            width=width,
            height=height,
            camera_type=CameraType.PERSPECTIVE,
        )

        camera_out = camera_out.to(self.device)

        return camera_out,width,height
    
    def render(self, xcr:np.ndarray, xpr:np.ndarray=None,
               visual_mode:Literal["static","dynamic"]="static"):
        
        if visual_mode == "static":
            image = self.static_render(xcr)
        elif visual_mode == "dynamic":
            image = self.dynamic_render(xcr,xpr)
        else:
            raise ValueError(f"Invalid visual mode: {visual_mode}")
        
        # Convert to numpy
        image = image.cpu().numpy()

        # Convert to uint8
        image = (255*image).astype(np.uint8)

        return image
        
    def static_render(self, xcr:np.ndarray) -> torch.Tensor:
        # Extract the pose
        T_c2n = pose2nerf_transform(np.hstack((xcr[0:3],xcr[6:10])))
        P_c2n = torch.tensor(T_c2n[0:3,:]).float()

        # Render from a single pose
        camera_to_world = P_c2n[None,:3, ...]
        self.camera_out.camera_to_worlds = camera_to_world

        # render outputs
        with torch.no_grad():
            outputs = self.pipeline.model.get_outputs_for_camera(self.camera_out, obb_box=None)

        image = outputs["rgb"]

        return image

    def dynamic_render(self, xcr:np.ndarray,xpr:np.ndarray,frames:int=10) -> torch.Tensor:
        # Get interpolated poses
        Xs = torch.zeros(xcr.shape[0],frames)
        for i in range(xcr.shape[0]):
            Xs[i,:] = torch.linspace(xcr[i], xpr[i], frames)
        
        # Make sure quaternion interpolation is correct
        xref = xcr
        for i in range(frames):
            Xs[6:10,i] = th.obedient_quaternion(Xs[6:10,i],xref)
            xref = Xs[:,i]

        # Render images
        images = []
        for i in range(frames):
            images.append(self.static_render(Xs[:,i]))

        # Average across images
        image = torch.mean(images, axis=0)

        return image
    
def get_nerf(map:str) -> NeRF:
    # Generate some useful paths
    workspace_path = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    main_dir_path = os.getcwd()
    nerf_dir_path = os.path.join(workspace_path,"nerf_data")
    maps = {
        "gate_left":"sv_917_3_left_nerfstudio",
        "gate_right":"sv_917_3_right_nerfstudio",
        "gate_mid":"sv_1007_gate_mid",
        "clutter":"sv_712_nerfstudio",
        "backroom":"backroom",
        "flightroom":"sv_1018_3",
        "SRC":"srcvid"
    }

    map_folder = os.path.join(nerf_dir_path,'outputs',maps[map])
    for root, _, files in os.walk(map_folder):
        if 'config.yml' in files:
            nerf_cfg_path = os.path.join(root, 'config.yml')

    # Go into NeRF data folder and get NeRF object (because the NeRF instantation
    # requires the current working directory to be the NeRF data folder)
    os.chdir(nerf_dir_path)
    nerf = NeRF(Path(nerf_cfg_path))
    os.chdir(main_dir_path)

    return nerf

def pose2nerf_transform(pose):

    # Realsense to Drone Frame
    T_r2d = np.array([
        [ 0.99250, -0.00866,  0.12186,  0.10000],
        [ 0.00446,  0.99938,  0.03463, -0.03100],
        [-0.12209, -0.03383,  0.99194, -0.01200],
        [ 0.00000,  0.00000,  0.00000,  1.00000]
    ])
    
    # Drone to Flightroom Frame
    T_d2f = np.eye(4)
    T_d2f[0:3,:] = np.hstack((R.from_quat(pose[3:]).as_matrix(),pose[0:3].reshape(-1,1)))

    # Flightroom Frame to NeRF world frame
    T_f2n = np.array([
        [ 1.000, 0.000, 0.000, 0.000],
        [ 0.000,-1.000, 0.000, 0.000],
        [ 0.000, 0.000,-1.000, 0.000],
        [ 0.000, 0.000, 0.000, 1.000]
    ])

    # Camera convention frame to realsense frame
    T_c2r = np.array([
        [ 0.0, 0.0,-1.0, 0.0],
        [ 1.0, 0.0, 0.0, 0.0],
        [ 0.0,-1.0, 0.0, 0.0],
        [ 0.0, 0.0, 0.0, 1.0]
    ])

    # Get image transform
    T_c2n = T_f2n@T_d2f@T_r2d@T_c2r

    return T_c2n