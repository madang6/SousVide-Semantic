
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSimSolver, AcadosSim
from dynamics.quadcopter_model import export_quadcopter_ode_model
import synthesize.solvers.min_snap as ms
import synthesize.trajectory_helper as th
import numpy as np
import torch
from casadi import vertcat
import scipy.linalg
from typing import List,Dict,Union,Tuple,Literal
import numpy.typing as npt
import time
from copy import deepcopy
import shutil
import os
import visualize.plot_synthesize as ps

class VehicleRateMPC():
    def __init__(self,
                 traj_config:Dict[str,Dict[str,Union[float,np.ndarray]]],
                 drone: Dict[str,Union[float,np.ndarray]],hz:int,
                 use_RTI:bool=False,name:str="policy",tpad:float=5.0) -> None:
        
        # Controller Configuration ================================================================
        
        # --------------------------------------------------------------------------------------------
                
        # # MoCap Weights --------------------------
        # Nhn = 40
        # Qk = np.diag([
        #     5.0e-1, 5.0e-1, 5.0e-1,
        #     5.0e-2, 5.0e-2, 5.0e-2,
        #     5.0e-2, 5.0e-2, 5.0e-2, 5.0e-2
        #     ])
        # Rk = np.diag([
        #     1e+0, 1e-1, 1e-1, 1e-1
        #     ])

        # QN = np.diag([
        #     5.0e-2, 5.0e-2, 5.0e-2,
        #     5.0e-2, 5.0e-2, 5.0e-2,
        #     5.0e-2, 5.0e-2, 5.0e-2, 5.0e-2
        #     ])
        
        # Backroom Weights --------------------------
        Nhn = 40
        Qk = np.diag([
            5.0e-1, 5.0e-1, 5.0e-1,
            1.0e-1, 1.0e-1, 1.0e-1,
            2.0e-1, 2.0e-1, 2.0e-1, 2.0e-1
            ])
        Rk = np.diag([
            1e+0, 1e-1, 1e-1, 1e-1
            ])

        QN = np.diag([
            5.0e-2, 5.0e-2, 5.0e-2,
            5.0e-2, 5.0e-2, 5.0e-2,
            5.0e-2, 5.0e-2, 5.0e-2, 5.0e-2
            ])
        
        # --------------------------------------------------------------------------------------------

        Ws_mpc = np.array([
                1e-0, 1e-0, 1e-0,
                1e-6, 1e-6, 1e-6,
                1e-6, 1e-6, 1e-6, 1e-6])

        # ==========================================================================================

        # Some useful intermediate variables
        nx,nu = drone["nx_br"], drone["nu_br"]
        ny,ny_e = nx+nu,nx
        tf = Nhn/hz
        lbu,ubu = drone["lbu"],drone["ubu"]
        solver_json = 'acados_ocp_nlp_'+name+'.json'
        
        # Pad the trajectory for consistent ending
        kff = list(traj_config["keyframes"])[-1]
        t_pd = traj_config["keyframes"][kff]["t"]+Nhn/hz+tpad
        fo_pd = np.array(traj_config["keyframes"][kff]["fo"])[:,0:3].tolist()

        traj_config_pd = deepcopy(traj_config)
        traj_config_pd["keyframes"]["fof"] = {
            "t":t_pd,
            "fo":fo_pd}

        # Solve Padded Trajectory
        output = ms.solve(traj_config_pd)
        if output is not False:
            Tpi, CPi = output
        else:
            raise ValueError("Padded trajectory (for VehicleRateMPC) not feasible. Aborting.")
        tXUi = th.ts_to_tXU(Tpi,CPi,drone,hz)
        
        # Setup OCP variables
        ocp = AcadosOcp()
        ocp.dims.N = Nhn

        ocp.model = export_quadcopter_ode_model(drone["m"],drone["tn"])        
        ocp.model.cost_y_expr = vertcat(ocp.model.x, ocp.model.u)
        ocp.model.cost_y_expr_e = ocp.model.x

        ocp.cost.cost_type = 'NONLINEAR_LS'
        ocp.cost.cost_type_e = 'NONLINEAR_LS'

        ocp.cost.W = scipy.linalg.block_diag(Qk,Rk)
        ocp.cost.W_e = QN

        ocp.cost.yref = np.zeros((ny,))
        ocp.cost.yref_e = np.zeros((ny_e, ))

        ocp.constraints.x0 = tXUi[1:11,0]
        ocp.constraints.lbu = lbu
        ocp.constraints.ubu = ubu
        ocp.constraints.idxbu = np.array([0, 1, 2, 3])

        ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
        ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
        ocp.solver_options.integrator_type = 'IRK'
        ocp.solver_options.sim_method_newton_iter = 10

        if use_RTI:
            ocp.solver_options.nlp_solver_type = 'SQP_RTI'
        else:
            ocp.solver_options.nlp_solver_type = 'SQP'

        ocp.solver_options.qp_solver_cond_N = Nhn
        ocp.solver_options.tf = tf
        ocp.solver_options.qp_solver_warm_start = 1
        
        # Controller Variables
        self.name = "VehicleRateMPC"
        self.Nx,self.Nu = nx,nu
        self.tXUi = tXUi                                                                        # Ideal Trajectory
        self.Qk,self.Rk,self.QN = Qk,Rk,QN                                                      # Cost Matrices
        self.lbu,self.ubu = lbu,ubu                                                             # Input Limits
        self.wts = Ws_mpc                                                              # Search weights for xv_ds
        self.ns = int(hz/5)                                                            # Search window size for xv_ds
        self.hz = hz                                                                   # Frequency of the MPC rollout
        self.use_RTI = use_RTI                                                                  # Use RTI flag
        self.model = ocp.model                                                                  # Acados OCP
        self.ocp_solver = AcadosOcpSolver(ocp,json_file=solver_json,verbose=False)            # Acados OCP Solver
        
        self.code_export_directory = ocp.code_export_directory
        self.solver_json_file = os.path.join(os.path.dirname(self.code_export_directory),solver_json)

        # # Warm Start
        # for i in range(Nhn):
        #     xws = self.tXUi[1:11,i]
        #     uws = np.hstack((
        #         -np.mean(self.tXUi[14:18,i],axis=0),
        #         self.tXUi[11:14,i]))
            
        #     self.ocp_solver.set(i, 'x', xws)
        #     self.ocp_solver.set(i, 'u', uws)

        # do some initial iterations to start with a good initial guess
        for _ in range(5):
            self.control(np.zeros(4),0.0,tXUi[1:11,0],np.zeros(10))

    def control(self,
                upr:np.ndarray,
                tcr:float,xcr:np.ndarray,
                obj:np.ndarray,
                icr:Union[npt.NDArray[np.uint8],None]=None,zcr:Union[torch.Tensor,None]=None) -> Tuple[
                    np.ndarray,None,None,np.ndarray]:
        
        # Unused arguments
        _ = upr,obj,icr,zcr
        
        # Start timer
        t0 = time.time()

        # Get reference trajectory
        yref = self.get_yref(xcr,tcr)

        # Set reference trajectory
        for i in range(self.ocp_solver.acados_ocp.dims.N):
            self.ocp_solver.cost_set(i, "yref", yref[:,i])
        self.ocp_solver.cost_set(self.ocp_solver.acados_ocp.dims.N, "yref", yref[0:10,-1])
        
        # # Set input limit
        # du0 = np.array([0.2,2.0,2.0,2.0])
        # lbu = np.clip(upr-du0,self.lbu,self.ubu)
        # ubu = np.clip(upr+du0,self.lbu,self.ubu)

        # self.ocp_solver.constraints_set(0, "lbu", lbu)
        # self.ocp_solver.constraints_set(0, "ubu", ubu)

        # duk = np.array([0.2,2.0,2.0,2.0])
        # for i in range(1, self.ocp_solver.acados_ocp.dims.N):
        #     lbu = np.clip(yref[10:14,i]-duk,self.lbu,self.ubu)
        #     ubu = np.clip(yref[10:14,i]+duk,self.lbu,self.ubu)

        #     self.ocp_solver.constraints_set(i, "lbu", lbu)
        #     self.ocp_solver.constraints_set(i, "ubu", ubu)

        t1 = time.time()
        if self.use_RTI:
            # preparation phase
            self.ocp_solver.options_set('rti_phase', 1)
            status = self.ocp_solver.solve()

            # set initial state
            self.ocp_solver.set(0, "lbx", xcr)
            self.ocp_solver.set(0, "ubx", xcr)

            # feedback phase
            self.ocp_solver.options_set('rti_phase', 2)
            status = self.ocp_solver.solve()

            ucc = self.ocp_solver.get(0, "u")
        else:

            # solve ocp and get next control input
            try:
                ucc = self.ocp_solver.solve_for_x0(x0_bar=xcr)
            except:
                print("Warning: VehicleRateMPC failed to solve OCP. Using previous input.")
                ucc = self.ocp_solver.get(0, "u")

        t2 = time.time()

        # End timer
        tsol = np.array([t1-t0,t2-t1,0.0,0.0])

        return ucc,None,None,tsol
    
    def get_yref(self,xcr:np.ndarray,ti:float) -> np.ndarray:
        # Get relevant portion of trajectory
        idx_i = int(self.hz*ti)
        ks0 = np.clip(idx_i-self.ns,0,self.tXUi.shape[1]-1)
        ksf = np.min([idx_i+self.ns,self.tXUi.shape[1]])
        xi = self.tXUi[1:11,ks0:ksf]

        # Find index of nearest state
        dx = xi-xcr.reshape(-1,1)
        idx0 = ks0 + np.argmin(self.wts.T@dx**2)
        idxf = idx0 + self.ocp_solver.acados_ocp.dims.N+1

        # Pad if idxf is greater than the last index
        if idxf < self.tXUi.shape[1]:
            xref = self.tXUi[1:11,idx0:idxf]
            
            ufref = -np.mean(self.tXUi[14:18,idx0:idxf],axis=0)
            uwref = self.tXUi[11:14,idx0:idxf]
            
            yref = np.vstack((
                xref,
                ufref,
                uwref))
        else:
            print("Warning: VehicleRateMPC.get_yref() padding trajectory. Increase your padding horizon.")
            xref = self.tXUi[1:11,idx0:]
            ufref = -np.mean(self.tXUi[14:18,idx0:],axis=0)
            uwref = self.tXUi[11:14,idx0:]

            yref = np.vstack((
                xref,
                ufref,
                uwref))
            
            yref = np.hstack((yref,np.tile(yref[:,-1:],(1,idxf-self.tXUi.shape[1]))))

        return yref
    
    def generate_simulator(self,hz):
        sim = AcadosSim()
        sim.model = self.model
        sim.solver_options.T = 1/hz
        sim.solver_options.integrator_type = 'IRK'

        sim_json = 'acados_sim_nlp.json'
        self.sim_json_file = os.path.join(os.path.dirname(self.code_export_directory),sim_json)

        return AcadosSimSolver(sim,json_file=sim_json,verbose=False)
    
    def clear_generated_code(self):
        try:
            os.remove(self.solver_json_file)
            shutil.rmtree(self.code_export_directory)
            os.remove(self.sim_json_file)
        except:
            pass