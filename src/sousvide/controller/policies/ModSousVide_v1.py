import torch
import numpy as np
from torch import nn
from controller.policies.ComponentNetworks import *
from controller.policies.SousVide_v1 import SousVide_v1
from typing import Dict,Tuple,Union,List,Any,Callable

class ModSousVide_v1(nn.Module):    
    def __init__(self,
                 father:SousVide_v1,
                 name:str="ModSousVide_v1"):
        

        # Initial Parent Call
        super(ModSousVide_v1,self).__init__()

        self.name = father.name
        self.Nz = father.Nz
        self.network = father.network
        self.get_data = father.get_data
        self.get_network = father.get_network

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
        ix_tx_vis  = self.network["VisionMLP"].ix_tx
        ix_tx_com  = self.network["CommanderSV"].ix_tx
        ix_obj_com = self.network["CommanderSV"].ix_obj

        # Extract Data
        xnn = {
            "tx_com" : tx[ix_tx_com].flatten(),
            "obj_com": Obj[ix_obj_com].flatten(),
            "dxu_par": DxU[ix_DxU_par[0],hy_idx+ix_DxU_par[1]].flatten(),
            "img_vis": Img.clone(),
            "tx_vis" : tx[ix_tx_vis].flatten()
        }

        return xnn

    def get_commander_inputs(self,xnn: Dict[str,torch.Tensor]) -> Tuple[torch.Tensor,...]:
        return (xnn["tx_com"],xnn["obj_com"],xnn["dxu_par"],xnn["img_vis"],xnn["tx_vis"])
    
    def get_parameter_data(self,xnn:Dict[str,torch.Tensor],ynn:Dict[str,torch.Tensor]) -> Tuple[Tuple[torch.Tensor],torch.Tensor]:
        return (xnn["dxu_par"],),ynn["mfn"]
    
    def get_commander_data(self,xnn: Dict[str,torch.Tensor],ynn: Dict[str,torch.Tensor]) -> Tuple[Tuple[torch.Tensor,...],torch.Tensor]:
        return self.get_commander_inputs(xnn),ynn["unn"]

    def get_image_data(self,xnn: Dict[str,torch.Tensor]) -> torch.Tensor:
        return xnn["img_vis"]
    
    def forward(self,tx_com:torch.Tensor,obj_com:torch.Tensor,
                dxu_par:torch.Tensor,
                img_vis:torch.Tensor,tx_vis:torch.Tensor) -> Tuple[torch.Tensor,None]:
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

        # Command Network
        y_vis,_ = self.network["VisionMLP"](img_vis,tx_vis)
        y_com,_ = self.network["CommanderSV"](tx_com,obj_com,z_par,y_vis)

        return y_com,None