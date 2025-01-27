import os
import torch

import sousvide.control.policies.ComponentNetworks as cn
from sousvide.control.policies.hpnet import HPNet
from sousvide.control.policies.svnet import SVNet
from sousvide.control.policies.svnet_no_rma import SVNetNoRMA
from sousvide.control.policies.svnet_no_pretrain import SVNetNoPreTrain
from sousvide.control.policies.svnet_direct import SVNetDirect
from sousvide.control.policies.AdvisorNetworks import SimpleAdvisor
from typing import Union,Tuple,Dict,Any

def policy_factory(path:str,config:Dict[str,Any],device:torch.device) -> Tuple[Union[
    HPNet,SVNet,SVNetNoRMA,SVNetNoPreTrain,SVNetNoPreTrain,SVNetDirect],int]:
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
        if policy_type == "HPNet":
            torch.save(HPNet(config),model_path)
        elif policy_type == "SVNet":
            torch.save(SVNet(config),model_path)
        elif policy_type == "SVNetNoRMA":
            torch.save(SVNetNoRMA(config),model_path)
        elif policy_type == "SVNetNoPreTrain":
            torch.save(SVNetNoPreTrain(config),model_path)
        elif policy_type == "SVNetDirect":
            torch.save(SVNetDirect(config),model_path)
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
