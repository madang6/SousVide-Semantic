import numpy as np
from typing import Dict,Union

def generate_preset_config(drone_config:Dict[str,Union[float,np.ndarray]]) -> Dict[str,Union[float,np.ndarray]]:
    """
    Generate a preset config (currently either Iris for Gazebo or Carl for the real world).
    
    Args:
    - drone_config: Preset drone config to generate.

    Returns:
    - config: Dictionary with the quadcopter configuration.
    """
    config_name = drone_config["name"]
    m = drone_config["mass"]
    Impp = np.array(drone_config["massless_inertia"])
    lf = np.array(drone_config["arm_front"])
    lb = np.array(drone_config["arm_back"])
    fn = drone_config["force_normalized"]
    tg = drone_config["torque_gain"]
    
    return generate_config(m,Impp,lf,lb,fn,tg,config_name)

def generate_config(m:float,Impp:np.ndarray,
                    lf:np.ndarray,
                    lb:np.ndarray,
                    fn:float,tg:float,
                    name:str='the_shepherd') -> Dict["str",Union[float,np.ndarray]]:
    """
    Generate a dictionary with the quadcopter configuration. The dictionary contains the following
    keys:

    Variable Constants:
    - m: Mass of the quadcopter (kg)
    - Impp: Massless Inertia tensor of the quadcopter (m^2)
    - lf: [x,y] distance from the center of mass to the front motors
    - lb: [x,y] distance from the center of mass to the back motors
    - fn: Normalized motor force gain
    - tG: Motor torque gain (after normalizing by fn)
    
    Fixed Constants:
    - nx_fs: Number of states for the full state model
    - nu_fs: Number of inputs for the full state model
    - nx_br: Number of states for the body rate model
    - nu_br: Number of inputs for the body rate model
    - nu_va: Number of inputs for the vehicle attitude model
    - lbu: Lower bound on the inputs
    - ubu: Upper bound on the inputs
    - tf: Time horizon for the MPC
    - hz: Frequency of the MPC
    - Qk: Stagewise State weight matrix for the MPC
    - Rk: Stagewise Input weight matrix for the MPC
    - QN: Terminal State weight matrix for the MPC
    - Ws: Search weights for the MPC (to get xv_ds)

    Derived Constants:
    - Iinv: Inverse of the inertia tensor
    - fMw: Matrix to convert from forces to moments
    - wMf: Matrix to convert from moments to forces
    - tn: Total normalized thrust

    Misc:
    - name: Name of the quadcopter

    The default values are for the Iris used in the Gazebo SITL simulation.
    
    """

    # Initialize the dictionary
    quad = {}
    
    # Variable Quadcopter Constants ===========================

    # F=ma, T=Ia Variables
    quad["m"],quad["I"] = m,np.diag(m*Impp)
    quad["lf"] = lf
    quad["lb"] = lb
    quad["fn"],quad["tg"] = fn, tg
    
    # Model Constants
    quad["nx_fs"],quad["nu_fs"] = 13,4
    quad["nx_br"],quad["nu_br"] = 10,4
    quad["nu_va"] = 5
    quad["lbu"] = np.array([-1.0, -5.0, -5.0, -5.0])
    quad["ubu"] = np.array([ 0.0,  5.0,  5.0,  5.0])
    
    # Derive Quadcopter Constants
    fMw = fn*np.array([
            [   -1.0,   -1.0,   -1.0,   -1.0],
            [ -lf[1],  lf[1],  lb[1], -lb[1]],
            [  lf[0], -lb[0],  lf[0], -lb[0]],
            [     tg,     tg,    -tg,    -tg]])
    
    quad["Iinv"] = np.diag(1/(m*Impp))
    quad["fMw"] = fMw
    quad["wMf"] = np.linalg.inv(fMw)
    quad["tn"] = quad["fn"]*quad["nu_fs"]

    # name
    quad["name"] = name
    
    return quad