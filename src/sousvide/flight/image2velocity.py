from typing import Tuple
import numpy as np
from scipy.spatial.transform import Rotation as R
import cv2

class Image2Velocity:
    def __init__(self):
        self.orb = cv2.ORB_create()
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.K = np.array([[462.956,   0.000, 323.076],
                            [ 0.000, 463.002, 181.184],
                            [ 0.000,   0.000,   1.000]])
        self.dv = 0.2

        self.t_cr,self.t_pr = None,None
        self.x_cr,self.x_pr = None,None
        self.kp_cr,self.kp_pr = None,None
        self.des_cr,self.des_pr = None,None
        self.depth_img_cr,self.depth_img_pr = None,None
    
    def update_keypoints_and_descriptors(self,rgb_cr,depth_img_cr,t_cr,x_cr):
        self.t_pr = self.t_cr
        self.x_pr = self.x_cr
        self.kp_pr = self.kp_cr
        self.des_pr = self.des_cr
        self.depth_img_pr = self.depth_img_cr

        self.t_cr = t_cr
        self.x_cr = x_cr
        self.kp_cr,self.des_cr = self.get_keypoints_and_descriptors(rgb_cr)

        self.depth_img_cr = depth_img_cr

    def get_keypoints_and_descriptors(self,rgb):
        return self.orb.detectAndCompute(rgb, None)
    
    def get_matches(self,threshold:float=None):
        matches = self.bf.match(self.des_pr, self.des_cr)

        if threshold is not None:
            matches = [match for match in matches if match.distance < threshold]

        return matches
    
    def get_features(self,matches):
        ft_pr = np.float32([self.kp_pr[m.queryIdx].pt for m in matches])
        ft_cr = np.float32([self.kp_cr[m.trainIdx].pt for m in matches])

        return ft_pr,ft_cr

    def get_points(self,features0,features1):
        Nft = features0.shape[0]
        points0,points1 = [],[]

        for i in range(Nft):
            px0, py0 = int(features0[i,0]), int(features0[i,1])
            px1, py1 = int(features1[i,0]), int(features1[i,1])

            if ((px1< self.depth_img_cr.shape[1]) and (px1 >= 0) and 
                (py1 < self.depth_img_cr.shape[0]) and (py1 >= 0)):

                depth0 = self.depth_img_pr[py0, px0] 
                depth1 = self.depth_img_cr[py1, px1]

                if (depth0 > 0) and (depth1 > 0):
                    point0 = [
                        (px0 - self.K[0, 2]) * depth0 / self.K[0, 0],
                        (py0 - self.K[1, 2]) * depth0 / self.K[1, 1],
                        depth0
                    ]

                    point1 = [
                        (px1 - self.K[0, 2]) * depth1 / self.K[0, 0],
                        (py1 - self.K[1, 2]) * depth1 / self.K[1, 1],
                        depth1
                    ]
                    
                    points0.append(point0)
                    points1.append(point1)

        return np.array(points0),np.array(points1)

    def compute_velocity(self) -> Tuple[np.ndarray,np.ndarray]:
        try:
            matches = self.get_matches()

            ft_pr,ft_cr = self.get_features(matches)
            pts_pr,pts_cr = self.get_points(ft_pr,ft_cr)

            R_bpr2w = R.from_quat(self.x_pr[6:10]).as_matrix()
            R_bcr2w = R.from_quat(self.x_cr[6:10]).as_matrix()

            pts_pr = R_bpr2w@pts_pr.T
            pts_cr = R_bcr2w@pts_cr.T

            vels = (pts_cr-pts_pr)/(self.t_cr-self.t_pr)

            mu_vel = np.mean(vels,axis=1)
            std_vel = np.std(vels,axis=1)
        except:
            mu_vel = np.array([None,None,None])
            std_vel = np.array([None,None,None])

        return mu_vel,std_vel
