import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from controller.pilot import Pilot
import torch
import instruct.synthesized_data as sd
import dynamics.quadcopter_config as qc
import imageio
import os
import json
import cv2
from matplotlib.colors import Normalize
from typing import Dict, Literal, Union, TypedDict, List
from controller import vr_mpc
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch.nn.functional as F

from captum.attr import (
    LayerGradCam,
    GuidedGradCam,
    GuidedBackprop,
    IntegratedGradients,
    LayerAttribution)

# Wrapper classes for captum to work with the single output model
class SingleVectorWrapper(nn.Module):
    def __init__(self, base_model):
        super(SingleVectorWrapper, self).__init__()
        self.base_model = base_model

    def forward(self, *x):
        out1, _ = self.base_model(*x)  # Use only the first output
        return out1  # Single output for captum
    
class SingleScalarWrapper(nn.Module):
    def __init__(self, base_model,course_name,drone_name,hz):
        super(SingleScalarWrapper, self).__init__()
        self.base_model = base_model

        workspace_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        with open(os.path.join(workspace_path,"configs","courses",course_name+".json")) as json_file:
            course_config = json.load(json_file)
        with open(os.path.join(workspace_path,"configs","drones",drone_name+".json")) as json_file:
            frame_config = json.load(json_file)
        drone_config = qc.generate_preset_config(frame_config)

        self.ctl = vr_mpc.VehicleRateMPC(course_config,drone_config,hz)
        self.tact,self.xact = np.zeros(0),np.zeros(10)
        
    def forward(self,*x):
        ucc,_,_,_ = self.ctl.control(None,self.tact,self.xact,None,None,None)
        ref = torch.tensor(ucc).float().unsqueeze(0).to(x[0].device)
        ref = ref.repeat(x[0].shape[0],1)

        cmd, _ = self.base_model(*x)
        output = F.cosine_similarity(cmd, ref)

        return  output
    
# Baseline Image
transform = A.Compose([                                             # Image transformation pipeline
                A.Resize(256, 256),
                A.CenterCrop(224, 224),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()
                ])

baseline_image_arr = np.ones((360,640,3),dtype=np.uint8)
baseline_image_arr[:,:,0] = 0
baseline_image_arr[:,:,1] = 255
baseline_image_arr[:,:,2] = 0
baseline_image = transform(image=baseline_image_arr)["image"]
# baseline_image = transform(image=np.random.rand(340, 640, 3) * 255)["image"]


def get_images_array(images_raw:torch.Tensor,
                     mean:np.ndarray=np.array([0.485, 0.456, 0.406]),
                     std:np.ndarray=np.array([0.229, 0.224, 0.225])) -> np.ndarray:
    images_prc = images_raw.cpu().detach().numpy()
    images_prc = images_prc.transpose(0,2,3,1)
    images_prc = np.stack(images_prc,axis=0)
    images_prc = images_prc*std[None, None,:] + mean[None, None,:]

    return images_prc

def generate_salience_maps(cohort_name:str,
                           width:int=672,height:int=672):

    # Get the trained pilots in the cohort
    workspace_path = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    cohort_dir = os.path.join(workspace_path, "cohorts", cohort_name)
    
    student_names = []
    for root, _, files in os.walk(cohort_dir):
        for file in files:
            if file.startswith("losses"):
                student_names.append(os.path.basename(root))
                break
    
    if len(student_names) == 0:
        raise ValueError("No trained pilots found in the cohort directory.")
    
    # Get the observation data
    observation_data_dir = os.path.join(cohort_dir, "rollout_data")
    observation_data = []
    for root, _, files in os.walk(observation_data_dir):
        for file in files:
            if not (file.startswith("images") or file.startswith("trajectories")):
                observation_data.append(os.path.join(root, file))
    observation_data.sort()

    # Outputs
    for student_name in student_names:
        # Load the pilot into a single output wrapper
        student = Pilot(cohort_name,student_name)

        # Setup network for saliency and choose target layer
        vModel = SingleVectorWrapper(student.model)
        sModel = SingleScalarWrapper(student.model,'track_gate_mid','carl',20)
        vModel.eval()
        sModel.eval()
        
        # Select Layer for GradCam
        last_layer = vModel.base_model.network["VisionMLP"].networks[0].networks.features[12]
        # target_layer = last_layer
        # target_layer = last_layer.expand3x3
        target_layer = last_layer.expand3x3_activation

        layer_gc = LayerGradCam(vModel, target_layer)
        int_grad = IntegratedGradients(sModel)

        for observation_file in observation_data[0:1]:
        # for observation_file in observation_data:
            # Get observation file name
            file_name = os.path.basename(observation_file).split(".")[0]

            # Load the data
            Tact,Xact  = sd.get_true_states(observation_file)
            inputs_raw = sd.get_input_data(observation_file,student)
            images_raw = sd.get_input_images(observation_file,student)

            # Generate baseline image
            int_grad_base = [torch.zeros_like(var).unsqueeze(0) for var in inputs_raw[0]]
            int_grad_base[3][0,:,:,:]= baseline_image

            # Initialize saliency dictionary
            Ndata = len(inputs_raw)
            saliencies = {"name":file_name,
                          "layer_gc":[],
                          "int_grad":[[] for _ in inputs_raw[0]]
                          }
            
            # Compute saliencies
            for idx,inputs in enumerate(inputs_raw):
                # Layered GradCam
                att_layer_gc = layer_gc.attribute(inputs=(*inputs,), target=None)
                att_layer_gc = LayerAttribution.interpolate(att_layer_gc,(224,224))
                att_layer_gc = att_layer_gc.squeeze().cpu().detach().numpy()

                saliencies["layer_gc"].append(att_layer_gc)

                # Integrated Gradients
                sModel.tact, sModel.xact = Tact[idx], Xact[:,idx]
                inputs_ig = [var.unsqueeze(0) for var in inputs]
                att_int_grad = int_grad.attribute(inputs=(*inputs_ig,), baselines=(*int_grad_base,))
                att_int_grad = [var.squeeze().cpu().detach().numpy() for var in att_int_grad]
                att_int_grad[3] = np.abs(att_int_grad[3])
                for i in range(len(att_int_grad)):
                    saliencies["int_grad"][i].append(att_int_grad[i])

            # Raw images
            images_prc = get_images_array(images_raw)

            # Heatmap configs
            cmap = plt.get_cmap("magma")         

            # Process Layer GradCam
            video_name = f"lgc_{saliencies['name']}.mp4"
            with imageio.get_writer(video_name, format='FFMPEG', mode='I', fps=20) as writer:
                for i in range(Ndata):
                    # Extract the saliency map
                    img_lgc = saliencies["layer_gc"][i]

                    # Normalize the saliency map wrt to the image
                    normalizer = Normalize(vmin=img_lgc.min(), vmax=img_lgc.max())
                    img_lgc = normalizer(img_lgc)

                    # Convert to heatmap
                    img_lgc = cmap(img_lgc)[:,:,:3]

                    # Combine the saliency map with the image
                    img_lgc = 0.5*images_prc[i] + 0.5*img_lgc

                    # Pack for video
                    img_lgc = (img_lgc*255).astype(np.uint8)
                    img_lgc = cv2.resize(img_lgc, (width, height), interpolation=cv2.INTER_CUBIC)
                    
                    writer.append_data(img_lgc)

            # Process Integrated Gradients
            video_name = f"ig_{saliencies['name']}.mp4"
            with imageio.get_writer(video_name, format='FFMPEG', mode='I', fps=20) as writer:
                for i in range(Ndata):
                    # Extract the saliency map
                    img_ig =  np.mean(saliencies["int_grad"][3][i],axis=0)

                    # Normalize the saliency map wrt to the image
                    normalizer = Normalize(vmin=img_ig.min(), vmax=img_ig.max())
                    img_ig = normalizer(img_ig)

                    # Convert to heatmap
                    img_ig = cmap(img_ig)[:,:,:3]

                    # Combine the saliency map with the image
                    img_ig = 0.15*images_prc[i] + 0.85*img_ig

                    # Pack for video
                    img_ig = (img_ig*255).astype(np.uint8)
                    img_ig = cv2.resize(img_ig, (width, height), interpolation=cv2.INTER_CUBIC)
                    
                    writer.append_data(img_ig)

        sModel.ctl.clear_generated_code()
            # print(layer_gc_arr.shape)
            # cmap = plt.get_cmap("magma")
            # print(img_layer_gc.shape)
            # norm = Normalize(vmin=img_layer_gc.min(), vmax=img_layer_gc.max())

            # heatmap = plt.get_cmap("magma")(saliencies["layer_gc"])
            # print(heatmap.shape)

    #         Saliencies.append(saliencies)
    #         Images_rgb.append(images_raw)

    # return Saliencies, Images_rgb

def salience2mp4_maps(Saliences:Dict[str,Union[np.ndarray,str]],images_raw:np.ndarray,idx:int=0,
                      width:int=672,height:int=672) -> None:
    sal_raw = Saliences["saliences"]
    sal_imgs = []
    for sal in sal_raw:
        sal_imgs.append(sal[3])

    sal = np.stack(sal_imgs, axis=0)
    mode = Saliences["mode"]
    Ndata = len(sal)

    # Normalize the salience maps
    norm = Normalize(vmin=sal.min(), vmax=sal.max())
    sal_norm = norm(sal)

    # Convert to grayscale
    weights = np.array([0.2989, 0.5870, 0.1140]).reshape(1, 3, 1, 1)
    gray_sal = np.sum(sal_norm * weights, axis=1)
    
    # Convert to heatmap
    heatmap = plt.get_cmap("magma")(gray_sal)[:,:,:,0:3]

    # Convert the image shape to match the heatmap
    images = images_raw.transpose(0,2,3,1)

    # Combine the salience maps with the images
    output = 0.6*heatmap + 0.4*images

    # Package for Video
    output = (output * 255).astype(np.uint8)

    # Save the video
    video_name = f"{mode}_{Saliences['name']}.mp4"
    with imageio.get_writer(video_name, format='FFMPEG', mode='I', fps=20) as writer:
        for i in range(Ndata):
            frame = output[i,:,:,:]
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_CUBIC)

            writer.append_data(frame)

    writer.close()
