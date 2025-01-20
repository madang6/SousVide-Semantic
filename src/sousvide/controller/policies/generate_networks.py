import os
import torch

import controller.policies.ComponentNetworks as cn
from controller.policies.HotPot_v0 import HotPot_v0
from controller.policies.DeepFry_v0 import DeepFry_v0
from controller.policies.SousVide_v0 import SousVide_v0
from controller.policies.SousVide_v1 import SousVide_v1
from controller.policies.SousVide_v1b import SousVide_v1b
from controller.policies.SousVide_v2 import SousVide_v2
from controller.policies.SousVide_v3 import SousVide_v3
from controller.policies.ModSousVide_v1 import ModSousVide_v1
from controller.policies.AdvisorNetworks import SimpleAdvisor
from typing import Union,Tuple,Dict,Any

def policy_factory(path:str,config:Dict[str,Any],device:torch.device) -> Tuple[Union[
    HotPot_v0,DeepFry_v0,
    SousVide_v0,SousVide_v1,SousVide_v1b,SousVide_v2,SousVide_v3,ModSousVide_v1],int]:
    """
    Factory function for creating a policy model.

    Args:
        path:   Path to the model.
        config: Configuration dictionary for the model. 

    Returns:y
        model: Policy model.    
    """

    # Some Useful Intermediate Variables
    model_path = os.path.join(path,"model.pth")

    # Check if model exists, if not create one.
    if os.path.isfile(model_path):
        pass
    else:
        # Create Folder/File
        os.makedirs(path,exist_ok=True)

        policy_type = config["type"]
        if policy_type == "HotPot_v0":
            torch.save(HotPot_v0(config),model_path)
        elif policy_type == "DeepFry_v0":
            torch.save(DeepFry_v0(config),model_path)
        elif policy_type == "SousVide_v0":
            torch.save(SousVide_v0(config),model_path)
        elif policy_type == "SousVide_v1":
            torch.save(SousVide_v1(config),model_path)
        elif policy_type == "SousVide_v1b":
            torch.save(SousVide_v1b(config),model_path)
        elif policy_type == "SousVide_v2":
            torch.save(SousVide_v2(config),model_path)
        elif policy_type == "SousVide_v3":
            torch.save(SousVide_v3(config),model_path)
        elif policy_type == "ModSousVide_v1":
            # Extract Iceman
            parts = model_path.split('/')
            parts[-2] = "Iceman"
            iceman_path = '/'.join(parts)
            iceman = torch.load(iceman_path)

            torch.save(ModSousVide_v1(iceman),model_path)
        else:
            raise ValueError(f"Policy '{policy_type}' not found.")

    # Load Model
    model = torch.load(model_path,map_location=device)

    # Extract feature vector size
    Nz = model.Nz

    return model,Nz

def advisor_factory(path:str,config:dict,device:torch.device) -> SimpleAdvisor:
    """
    Factory function for creating a policy model.

    Args:
        path:   Path to the model.
        config: Configuration dictionary for the model. 

    Returns:
        model: Policy model.    
    """

    # Some Useful Intermediate Variables
    model_path = os.path.join(path,"advisor.pth")

    # Check if model exists, if not create one.
    if os.path.isfile(model_path):
        pass
    else:
        # Create Folder/File
        os.makedirs(path,exist_ok=True)

        # Create Model
        torch.save(SimpleAdvisor(config),model_path)

    # Load Model
    model = torch.load(model_path,map_location=device)
    
    return model    
