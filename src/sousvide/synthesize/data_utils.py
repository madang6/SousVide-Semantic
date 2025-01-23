import numpy as np
import json
import os
import torch
from typing import Dict,Union,Tuple,List

import sousvide.synthesize.synthesize_helper as sh

from PIL import Image
from io import BytesIO
def decompress_data(image_dict:Dict[str,Union[str,np.ndarray]]) -> Dict[str,Union[str,np.ndarray]]:
    """
    We apply a compression if the images are too large to be saved in the .pt file. This function
    decompresses the images back to their original form.
    """
    assert 'images' in image_dict, "No images found in data dictionary"
    # Check if the image are processed or not. We do this by checking the array
    # order. If the order is (N, C, H, W) then the images are processed. If the
    # order is (N, H, W, C) then the images are unprocessed. We only compress if
    # the images are processed.
    if type(image_dict['images'][0]) == bytes:
        prc_imgs = image_dict['images']
        raw_imgs = []
        for frame_idx in range(len(prc_imgs)):
            # Process each image here
            imgpil = Image.open(BytesIO(prc_imgs[frame_idx]))
            raw_imgs.append(np.array(imgpil))

        image_dict['images'] = np.stack(raw_imgs,axis=0)
    else:
        pass

    return image_dict

def compress_data(Images):
    """
    We apply a compression if the images are too large to be saved in the .pt file. This function
    compresses the images to a smaller size.
    """
    assert 'images' in Images[0], "No images found in data dictionary"

    for image_dict in Images:
        # Check if the image are processed or not. We do this by checking the array
        # order. If the order is (N, C, H, W) then the images are processed. If the
        # order is (N, H, W, C) then the images are unprocessed. We only compress if
        # the images are processed.

        if image_dict['images'].shape[-1] == 3:
            raw_imgs = image_dict['images']
            prc_imgs = []
            for frame_idx in range(raw_imgs.shape[0]):
                img_arr = raw_imgs[frame_idx]
                imgpil = Image.fromarray(img_arr)

                buffer = BytesIO()
                imgpil.save(buffer, format='PNG')   
                buffer.seek(0)
                prc_imgs.append(buffer.getvalue())

            image_dict['images'] = prc_imgs
        else:
            pass
    
    return Images

def save_rollouts(cohort_path:str,course_name:str,
                  Trajectories:List[Tuple[np.ndarray,np.ndarray,np.ndarray]],
                  Images:List[torch.Tensor],
                  tXUi:np.ndarray,
                  stack_id:Union[str,int]) -> None:
    """
    Saves the rollout data to a .pt file in folders corresponding to coursename within the cohort 
    directory. The rollout data is stored as a list of rollout dictionaries of size stack_size for
    ease of comprehension and loading (at a cost of storage space).
    
    Args:
        cohort_path:    Cohort path.
        course_name:    Name of the course.
        Trajectories:   Rollout data.
        Images:         Image data.
        tXUi:           Ideal trajectory data.
        stack_id:       Stack id.

    Returns:
        None:           (rollout data saved to cohort directory)
    """

    # Create rollout course directory (if it does not exist)
    rollout_course_path = os.path.join(cohort_path,"rollout_data",course_name)
    if not os.path.exists(rollout_course_path):
        os.makedirs(rollout_course_path)
    
    # Save the stacks
    Ndata = sum([trajectory["Ndata"] for trajectory in Trajectories])
    data_set_name = str(stack_id).zfill(3) if type(stack_id) == int else str(stack_id)
    trajectory_data_set_path = os.path.join(rollout_course_path,"trajectories"+data_set_name+".pt")
    image_data_set_path = os.path.join(rollout_course_path,"images"+data_set_name+".pt")

    # Compress the image data
    Images = compress_data(Images)

    trajectory_data_set = {"data":Trajectories,
                           "tXUi":tXUi,
                            "set":data_set_name,"Ndata":Ndata,"course":course_name}
    image_data_set = {"data":Images,
                        "set":data_set_name,"Ndata":Ndata,"course":course_name}

    torch.save(trajectory_data_set,trajectory_data_set_path)
    torch.save(image_data_set,image_data_set_path)

def save_observations(cohort_path:str,course_name:str,
                      pilot_name:str,
                      observations:Dict[str,Union[np.ndarray,int,str]]) -> None:
    
    """
    Saves the observation data to a .pt file in folders corresponding to pilot name within the course
    directory within the cohort directory.

    Args:
        cohort_path:    Cohort path.
        course_name:    Name of the course.
        pilot_name:     Name of the pilot.
        observations:   Observation data.

    Returns:
        None:           (observation data saved to cohort directory)
    """

    # Create observation course directory (if it does not exist)
    observation_pilot_path = os.path.join(cohort_path,"observation_data",pilot_name)
    if not os.path.exists(observation_pilot_path):
        os.makedirs(observation_pilot_path)

    # Save the observations
    observations_data_path = os.path.join(
        observation_pilot_path,"observations_"+
        course_name+str(observations["set"])+".pt")

    torch.save(observations,observations_data_path)

# def flight2rollout_data(cohort_name:str,drone_name:str):
#     # Some useful path(s)
#     workspace_path = os.path.dirname(
#         os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

#     # Load drone and method configs
#     cohort_path = os.path.join(workspace_path,"cohorts",cohort_name)
#     flight_data_path = os.path.join(cohort_path,"flight_data")
#     drone_path  = os.path.join(workspace_path,"configs","drones",drone_name+".json")

#     with open(drone_path) as json_file:
#         drone_config = json.load(json_file)

#     # Gather all the flight data files
#     file_list = os.listdir(flight_data_path)
#     file_list = [os.path.join(flight_data_path, f) for f in file_list if f.endswith('.pt')]

#     # Assume the drone is Carl
#     drone = qc.generate_preset_config(drone_config)

#     # Convert flight data to training data
#     data_names = {}
#     for idx,file in enumerate(file_list):
#         # Load the flight data
#         data = torch.load(file)

#         # Check if the data is in the old format
#         data = flightdata_check(data)

#         # Extract Some Useful Intermediate Variables
#         parts = file.split("/")[-1].split("_")
#         file_name_components = [part for part in parts if not part[0].isdigit()]
#         prefix = file_name_components[0]
#         course_name = "_".join(file_name_components[1:-1])

#         # Check if course is in the dictionary
#         data_name = "_".join((prefix,course_name))
#         if data_name not in data_names:
#             data_names[data_name] = 0
#         else:
#             data_names[data_name] += 1

#         # Generate ideal trajectory
#         course_path = os.path.join(workspace_path,"configs","courses",course_name+".json")
#         with open(course_path) as json_file:
#             course_config = json.load(json_file)

#         Tpi,CPi = ms.solve(course_config)
#         tXUi = th.ts_to_tXU(Tpi,CPi,drone)
        
#         # Create save directory
#         save_path = os.path.join(cohort_path,"rollout_data",course_name)
#         if not os.path.exists(save_path):
#             os.makedirs(save_path)

#         # Pack into a dictionary
#         trajectory = {
#             "Tro":data["Tact"],"Xro":data["Xest"],"Uro":data["Uact"],
#             "Xid":None,"obj":data["obj"],"Ndata":data["Tact"].shape[0],"Tsol":data["Tsol"],"Adv":data["Adv"],
#             "rollout_id":str(data_names[data_name]).zfill(5),
#             "course":course_name,
#             "drone":drone
#             }

#         images = {
#             "images": data["Imgs"].numpy(),  # Reshape to (w, h, c)
#             "rollout_id":str(data_names[data_name]).zfill(5),"course":course_name
#         }

#         # Save the training data.
#         save_name = "_".join(("",prefix,course_name,str(data_names[data_name]+1).zfill(3)))
#         save_rollouts(cohort_name,course_name,[trajectory],[images],tXUi,save_name)

#         # # Print some diagnostics
#         # print("Generated ",data["Tact"].shape[0]," points of data for",course_name)

def flightdata_check(data:dict):
    # Keys reference
    new_keys = ['Imgs', 'Tact', 'Uact', 'Xref', 'Uref', 'Xest', 'Xext', 'Adv', 'Tsol', 'obj', 'n_im']
    old_keys = ['Tact', 'Xref', 'Uref', 'Xact', 'Uact', 'Adv', 'Tsol', 'Imgs', 'tXds', 'n_im']

    # Load the flight data
    data_keys = list(data.keys())

    if data_keys == new_keys:
        return data
    elif data_keys == old_keys:
        new_data = {
            'Imgs': data['Imgs'],
            'Tact': data['Tact'],
            'Uact': data['Uact'],
            'Xref': data['Xref'],
            'Uref': data['Uref'],
            'Xest': data['Xact'],
            'Xext': data['Xact'],
            'Adv': data['Adv'],
            'Tsol': data['Tsol'],
            'obj': sh.tXU_to_obj(data['tXds']),
            'n_im': data['n_im']
        }
        return new_data
    else:
        print("Data keys do not match expected keys")
        return None