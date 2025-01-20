import numpy as np
from scipy.spatial.transform import Rotation as R
import synthesize.trajectory_helper as th
from typing import Dict,Union,Tuple
import time

class VehicleAttitudePD():
    def __init__(self,
                 traj:Union[np.ndarray,Tuple[np.ndarray,np.ndarray]],
                 quad: Dict["str",Union[float,np.ndarray]]
                ):

        # Some useful intermediate variables
        if isinstance(traj,Tuple):
            Tpi,CPi = traj[0],traj[1]
            tXUi = th.ts_to_tXU(Tpi,CPi,quad)
        elif isinstance(traj,np.ndarray):
            tXUi = traj
        else:
            raise ValueError("Invalid trajectory format.")
    
        # Controller Variables
        self.Kp = quad["Kp"]                    # Position error gains
        self.Kv = quad["Kv"]                    # Velocity error gains
        self.m = quad["m"]                      # Mass
        self.tn = quad["tn"]                    # Total thrust gain
        self.kla = int(quad["tla"]*quad["hz"])  # Lookahead index
        self.tXUi = tXUi                        # Ideal Trajectory
        self.wts = quad["Ws_vas"]               # Search weights for xv_ds
        self.ns = int(quad["hz"]/2)             # Search window size for xv_ds
    
    def control(self,xv_cr:np.ndarray,ti:float=None) -> np.ndarray:
        # Start timer
        t_start = time.time()

        # Get rough index of nearest state
        if ti is None:
            idx_i = None
        else:
            idx_i = int(ti/(self.tXUi[0,1]-self.tXUi[0,0]))

        # Get desired state
        xv_ds = self.get_xv_ds(xv_cr,idx_i)

        # Some useful intermediate variables
        del_p = xv_ds[0:3]-xv_cr[0:3]
        del_v = xv_ds[3:6]-xv_cr[3:6]
        
        R_ds = R.from_quat(xv_ds[6:10]).as_matrix()
        R_cr = R.from_quat(xv_cr[6:10]).as_matrix()
        z_cr = R_cr[:,2]

        Fdes = -self.Kp@del_p - self.Kv@del_v + self.m*np.array([0,0,9.81])

        # Generate z_cmd vector
        z_cmd = Fdes/np.linalg.norm(Fdes)
        y_cmd = np.cross(z_cmd,R_ds[:,0])/np.linalg.norm(np.cross(z_cmd,R_ds[:,0]))
        x_cmd = np.cross(y_cmd,z_cmd)

        R_cmd = np.hstack((x_cmd.reshape(-1,1),y_cmd.reshape(-1,1),z_cmd.reshape(-1,1)))

        # Generate Thrust Command
        thrust = -np.dot(Fdes,z_cr)
        n_thrust = thrust/self.tn

        # Generate Quaternion Command
        q_cmd = R.from_matrix(R_cmd).as_quat()

        # vas command
        u_cmd = np.array([n_thrust,q_cmd[3],q_cmd[0],q_cmd[1],q_cmd[2]])

        # End timer
        t_sol = time.time()-t_start

        return u_cmd,t_sol
    
    def get_xv_ds(self,xv_cr:np.ndarray,idx_i=None) -> np.ndarray:
        # Get relevant portion of trajectory
        if idx_i is not None:
            ks0 = np.max([0,idx_i-self.ns])
            ksf = np.min([idx_i+self.ns,self.tXUi.shape[1]])

            Xs = self.tXUi[1:14,ks0:ksf]
        else:
            ks0 = 0
            Xs = self.tXUi[1:14,:]
        
        # Find the index of the lookahead state
        dxv = Xs-xv_cr.reshape(-1,1)
        idx_la = ks0+np.argmin(self.wts.T@dxv**2)+self.kla

        # Make sure the index is within bounds
        idx_la = np.min([idx_la,self.tXUi.shape[1]-1])

        return self.tXUi[1:14,idx_la]
