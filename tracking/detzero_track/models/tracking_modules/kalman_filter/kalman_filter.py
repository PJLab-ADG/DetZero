import copy

import numpy as np


class BaseKalmanFilter:
    """
    Base Kalman Filter
    """
    def __init__(self, bbox, name, score, frame_id, track_id, num_points=0,
                 x_dim=5, z_dim=3, delta_t=0.1, p=[1, 1], q=[1, 1], r=1):

        self.x_dim = x_dim
        self.z_dim = z_dim

        self.size = bbox[3:6]
        self.heading = bbox[6]
        self.name = name
        self.score = score
        self.update_score = score
        self.num_points = num_points

        self.frame_id = frame_id
        self.delta_t = delta_t

        self.track_id = track_id
        '''
        init filter parameters
        '''
        # state: [x, y, z, vx, vy]
        self.x = np.zeros((x_dim, 1), dtype=np.float32)
        self.x[:z_dim, :] = copy.deepcopy(bbox[:3].reshape(3, 1))

        self.bbox = np.zeros((bbox.shape[0]+2), dtype=np.float32)
        self.bbox[:bbox.shape[0]] = copy.deepcopy(bbox)

        self.F = np.eye(x_dim, dtype=np.float32)
        self.P = np.eye(x_dim, dtype=np.float32)
        self.Q = np.eye(x_dim, dtype=np.float32)

        self.H = np.eye(z_dim, x_dim, dtype=np.float32)
        self.R = np.eye(z_dim, dtype=np.float32)

        self.F[:2, 3:5] = np.eye(self.x_dim-self.z_dim, dtype=np.float32) * self.delta_t

        self.P[:self.z_dim, :self.z_dim] = self.P[:self.z_dim, :self.z_dim]*p[0]
        self.P[self.z_dim:, self.z_dim:] = self.P[self.z_dim:, self.z_dim:]*p[1]

        # meas: [x, y, z]
        self.Q[:3, :3] = self.Q[:3, :3]*q[0]
        self.Q[3:, 3:] = self.Q[3:, 3:]*q[1]
        self.R[:3, :3] = self.R[:3, :3]*r

        # update count
        self.hit = 1
        self.miss = 0

    def state(self):
        return self.x

    def info(self):
        data = {
            'boxes_global': self.bbox,
            'name': self.name,
            'score': self.score,
            'sample_idx': self.frame_id,
            'hit': self.hit,
            'num_points': self.num_points,
            'obj_ids': self.track_id,
        }

        return {self.track_id: data}


class KalmanFilter(BaseKalmanFilter):
    """
    not update object center and size
    """
    def __init__(self, bbox, name, score, frame_id, track_id, num_points=0,
                 x_dim=5, z_dim=3, delta_t=0.1, p=[1, 1], q=[1, 1], r=1, **kwargs):
        super().__init__(bbox, name, score, frame_id, track_id, num_points=num_points,
                 x_dim=x_dim, z_dim=z_dim, delta_t=delta_t, p=p, q=q, r=r)


    def predict(self, frame_id):
        """
        Preict one step
        """
        self.frame_id = frame_id

        temp_x = copy.deepcopy(self.x)
        if self.name == "Vehicle":
            speed_norm = np.linalg.norm(temp_x[self.z_dim:])
            if speed_norm <= np.max(self.size)/2.:
                temp_x[self.z_dim:] = 0.

        self.x = self.F @ temp_x
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.Q = self.Q * 1.5

        self.miss += 1
        self.hit = 0

        bbox = np.concatenate((self.x.reshape(-1)[:3], self.size, self.heading.reshape(-1), 
                               self.x.reshape(-1)[3:5]), axis=0)
        self.bbox = bbox

        return self.bbox

    def update(self, bbox, name, score, num_points, two_stage=False):
        """
        Update state with measurement
        """
        self.hit = 1
        self.miss = 0

        self.score = score
        self.num_points = num_points

        if two_stage:
            self.hit = 2
            return self.bbox 

        self.name = name
        self.update_score = max(score, 0.03)

        z = bbox[:3].reshape(3, 1)
        self.size = bbox[3:6]
        self.heading = bbox[6]

        res_z = z - self.H @ self.x

        r = 1.
        S = self.H @ self.P @ self.H.T + self.R * r
        inv_S = np.linalg.inv(S)

        K = self.P @ self.H.T @ inv_S
        self.x = self.x + K @ res_z
        self.P = self.P - K @ self.H @ self.P

        self.x[:3]  = copy.deepcopy(z)
        bbox = np.concatenate((bbox[0:3], self.size, self.heading.reshape(-1), 
                               self.x.reshape(-1)[3:5]), axis=0)
        self.bbox = bbox

        return self.bbox
