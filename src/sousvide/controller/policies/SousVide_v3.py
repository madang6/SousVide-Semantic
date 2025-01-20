import torch
import numpy as np
from torch import nn
from controller.policies.ComponentNetworks import *
from typing import Dict,Tuple,Union,List,Any,Callable

class SousVide_v3(nn.Module):    
    def __init__(self,config:Dict[str,Any],
                 name:str="SousVide_v3"):
        """
        Defines a pilot's neural network model.

        Network Description:
        Uses three separately trained networks.
        - Parameter: History Data -> Drone Parameters Decoder
        - Odometry:  Vision Data -> Odometry Data
        - Commander: Odometry Data + Drone Parameters Encoder + Current Data + Objective Data -> Command

        Args:
            config:         Configuration dictionary for the model.
            name:           Name of the policy.

        Variables:
            name:           Name of the policy.
            Nz:             Feature vector size.
            network:        Dictionary of the network components.
            get_data:       Dictionary of functions to extract data for the network.
            get_network:    Dictionary of network components for training.
        """

        # Initial Parent Call
        super(SousVide_v3,self).__init__()

        # ----------------------------------------------------------------------------------------------
        # Class Intermediate Variables
        # ----------------------------------------------------------------------------------------------

        history_enc_cfg = config["networks"][0]
        vision_enc_cfg  = config["networks"][1]
        command_sv2_cfg = config["networks"][2]

        history_enc_network = HistoryEncoder(
                history_enc_cfg["delta"],
                history_enc_cfg["frames"],
                history_enc_cfg["hidden_sizes"],
                history_enc_cfg["encoder_size"],
                history_enc_cfg["decoder_size"]
            )
        vision_enc_network = VisionEncoder(
                vision_enc_cfg["state"],
                vision_enc_cfg["state_layer"],
                vision_enc_cfg["hidden_sizes"],
                vision_enc_cfg["encoder_size"],
                vision_enc_cfg["decoder_size"],
                vision_enc_cfg["CNN"]
            )
        commander_sv2_network = CommandSV2(
                command_sv2_cfg["state"],
                command_sv2_cfg["feature"],
                command_sv2_cfg["frames"],
                command_sv2_cfg["objective"],
                history_enc_cfg["encoder_size"],
                command_sv2_cfg["hidden_sizes"],
                command_sv2_cfg["output_size"]
            )

        # ----------------------------------------------------------------------------------------------
        # Class Name Variables
        # ----------------------------------------------------------------------------------------------

        # Network ID
        self.name = name
        self.Nz = vision_enc_cfg["encoder_size"]

        # Network Components
        self.network = nn.ModuleDict({
            "HistoryEncoder": history_enc_network,
            "VisionEncoder": vision_enc_network,
            "CommanderSV2": commander_sv2_network
        })

        # Network Callers [Parameter,Odometry,Commander]
        self.get_data:Dict[str,Callable] = {                         
            "Parameter": self.get_parameter_data,
            "Odometry" : self.get_odometry_data,
            "Commander": self.get_commander_data,
            "Images": self.get_image_data
        }

        self.get_network:Dict[str,Dict[str,nn.Module]] = {
            "Parameter": {
                "Train" : self.network["HistoryEncoder"],
                "Unlock": self.network["HistoryEncoder"]
            },
            "Odometry": {
                "Train" : self.network["VisionEncoder"],
                "Unlock": self.network["VisionEncoder"]
            },
            "Commander": {
                "Train" :self,
                "Unlock":self.network["CommanderSV2"]
            }
        }
    
    def extract_inputs(self,tx:torch.Tensor,Obj:torch.Tensor,
                       Img:torch.Tensor,DxU:torch.Tensor,Znn:torch.Tensor,hy_idx:int) -> Dict[str,torch.Tensor]:
        """
        Takes the basis input variables and extracts the relevant inputs for the model. If the
        model component does not exist, the input is set to None.

        Args:
            tx:     Current state.
            Obj:    Objective.
            Img:    Image.
            DxU:    History deltas.
            Vio:    Odometry.
            hy_idx: History indices.

        Returns:
            xnn:    Extracted inputs for the model

        """

        # Unused Variables
        _ = Znn

        # Extract Indices
        ix_DxU_par = self.network["HistoryEncoder"].ix_DxU
        ix_tx_vis  = self.network["VisionEncoder"].ix_tx
        ix_tx_com  = self.network["CommanderSV2"].ix_tx
        ix_obj_com = self.network["CommanderSV2"].ix_obj
        ix_z_com   = self.network["CommanderSV2"].ix_z

        # Extract Data
        xnn = {
            "tx_com" : tx[ix_tx_com].flatten(),
            "obj_com": Obj[ix_obj_com].flatten(),
            "z_com"  : Znn[ix_z_com[0],hy_idx+ix_z_com[1]].flatten(),
            "dxu_par": DxU[ix_DxU_par[0],hy_idx+ix_DxU_par[1]].flatten(),
            "img_odo": Img.clone(),
            "tx_odo" : tx[ix_tx_vis].flatten(),
        }

        return xnn
    
    def get_commander_inputs(self,xnn: Dict[str,torch.Tensor]) -> Tuple[torch.Tensor,...]:
        return (xnn["tx_com"],xnn["obj_com"],xnn["z_com"],xnn["dxu_par"],xnn["img_odo"],xnn["tx_odo"])
    
    def get_odometry_data(self,xnn:Dict[str,torch.Tensor],ynn:Dict[str,torch.Tensor]) -> Tuple[Tuple[torch.Tensor,...],torch.Tensor]:
        ix_onn = np.array([0,1])
        return (xnn["img_odo"],xnn["tx_odo"]),ynn["onn"][ix_onn]
    
    def get_parameter_data(self,xnn:Dict[str,torch.Tensor],ynn:Dict[str,torch.Tensor]) -> Tuple[Tuple[torch.Tensor],torch.Tensor]:
        return (xnn["dxu_par"],),ynn["mfn"]
    
    def get_commander_data(self,xnn: Dict[str,torch.Tensor],ynn: Dict[str,torch.Tensor]) -> Tuple[Tuple[torch.Tensor,...],torch.Tensor]:
        return self.get_commander_inputs(xnn),ynn["unn"]
    
    def get_image_data(self,xnn: Dict[str,torch.Tensor]) -> torch.Tensor:
        return xnn["img_odo"]
    
    def forward(self,tx_com:torch.Tensor,obj_com:torch.Tensor,z_com:torch.Tensor,
                dxu_par:torch.Tensor,
                img_odo:torch.Tensor,tx_odo:torch.Tensor) -> Tuple[torch.Tensor,None]:
        """
        Defines the forward pass of the neural network model.

        Args:
            tx_com:     Current state for commander network.
            obj_com:    Objective for commander network.
            dxu_par:    History deltas for parameter network.
            img_vis:    Image for vision network.
            tx_vis:     Current state for vision network.

        Returns:
            ycm: Command network output.
            zcm: Empty.
        """

        # Parameter Network
        _,z_par = self.network["HistoryEncoder"](dxu_par)

        # Odometry Network
        _,z_odo = self.network["VisionEncoder"](img_odo,tx_odo)

        # Command Network
        y_com,_ = self.network["CommanderSV2"](tx_com,obj_com,z_par,z_odo,z_com)

        return y_com,z_odo