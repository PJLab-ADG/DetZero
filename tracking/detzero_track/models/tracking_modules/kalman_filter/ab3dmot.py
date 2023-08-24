import copy
import numpy as np

from filterpy.kalman import KalmanFilter

from .kalman_filter import BaseKalmanFilter


class AB3DMOT(BaseKalmanFilter):
    """
    Kalman Filter same with AB3DMOT
    """
    def __init__(self, bbox, name, score, frame_id, track_id, num_points=0,
                 x_dim=5, z_dim=3, delta_t=0.1, p=[1, 1], q=[1, 1], r=1):
        self.name = name
        self.score = score
        self.update_score = score
        self.num_points = -1

        self.frame_id = frame_id
        self.track_id = track_id

        self.bbox = np.zeros((bbox.shape[0]+2), dtype=np.float32)
        self.bbox[:bbox.shape[0]] = copy.deepcopy(bbox)

        # define constant velocity model
        self.kf = KalmanFilter(dim_x=10, dim_z=7)
        # x y z l w h theta vx vy vz
        self.kf.F = np.array([[1,0,0,0,0,0,0,1,0,0],      # state transition matrix
                              [0,1,0,0,0,0,0,0,1,0],
                              [0,0,1,0,0,0,0,0,0,1],
                              [0,0,0,1,0,0,0,0,0,0],  
                              [0,0,0,0,1,0,0,0,0,0],
                              [0,0,0,0,0,1,0,0,0,0],
                              [0,0,0,0,0,0,1,0,0,0],
                              [0,0,0,0,0,0,0,1,0,0],
                              [0,0,0,0,0,0,0,0,1,0],
                              [0,0,0,0,0,0,0,0,0,1]])     

        self.kf.H = np.array([[1,0,0,0,0,0,0,0,0,0],      # measurement function,
                              [0,1,0,0,0,0,0,0,0,0],
                              [0,0,1,0,0,0,0,0,0,0],
                              [0,0,0,1,0,0,0,0,0,0],
                              [0,0,0,0,1,0,0,0,0,0],
                              [0,0,0,0,0,1,0,0,0,0],
                              [0,0,0,0,0,0,1,0,0,0]])

        # # with angular velocity
        # self.kf = KalmanFilter(dim_x=11, dim_z=7)       
        # self.kf.F = np.array([[1,0,0,0,0,0,0,1,0,0,0],      # state transition matrix
        #                       [0,1,0,0,0,0,0,0,1,0,0],
        #                       [0,0,1,0,0,0,0,0,0,1,0],
        #                       [0,0,0,1,0,0,0,0,0,0,1],  
        #                       [0,0,0,0,1,0,0,0,0,0,0],
        #                       [0,0,0,0,0,1,0,0,0,0,0],
        #                       [0,0,0,0,0,0,1,0,0,0,0],
        #                       [0,0,0,0,0,0,0,1,0,0,0],
        #                       [0,0,0,0,0,0,0,0,1,0,0],
        #                       [0,0,0,0,0,0,0,0,0,1,0],
        #                       [0,0,0,0,0,0,0,0,0,0,1]])     

        # self.kf.H = np.array([[1,0,0,0,0,0,0,0,0,0,0],      # measurement function,
        #                       [0,1,0,0,0,0,0,0,0,0,0],
        #                       [0,0,1,0,0,0,0,0,0,0,0],
        #                       [0,0,0,1,0,0,0,0,0,0,0],
        #                       [0,0,0,0,1,0,0,0,0,0,0],
        #                       [0,0,0,0,0,1,0,0,0,0,0],
        #                       [0,0,0,0,0,0,1,0,0,0,0]])


        # self.kf.R[0:,0:] *= 10.   # measurement uncertainty
        self.kf.P[7:, 7:] *= 1000.     # state uncertainty, give high uncertainty to the unobservable initial velocities, covariance matrix
        self.kf.P *= 10.

        # self.kf.Q[-1,-1] *= 0.01    # process uncertainty
        self.kf.Q[7:, 7:] *= 0.01
        self.kf.x[:7] = bbox.reshape((7, 1))

        self.time_since_update = 0
        self.history = []
        self.hits = 1           # number of total hits including the first detection
        self.hit = 1
        self.miss = 0


        self.age = 0

    def predict(self, frame_id):
        """
        Preict one step
        """
        self.frame_id = frame_id

        self.kf.predict()      
        if self.kf.x[6] >= np.pi: self.kf.x[6] -= np.pi * 2
        if self.kf.x[6] < -np.pi: self.kf.x[6] += np.pi * 2

        self.age += 1
        if (self.time_since_update > 0):
            self.hit_streak = 0
            self.still_first = False
        self.time_since_update += 1

        self.miss += 1
        self.hit = 0

        bbox = self.kf.x.reshape(-1)[:9]
        self.bbox = bbox
        return self.bbox

    def update(self, bbox, name, score, num_points, two_stage=False):
        """ 
        Updates the state vector with observed bbox.
        """
        self.hit = 1
        self.miss = 0
        self.hits += 1
        self.score = score
        self.name = name

        ######################### orientation correction
        if self.kf.x[6] >= np.pi: self.kf.x[6] -= np.pi * 2    # make the theta still in the range
        if self.kf.x[6] < -np.pi: self.kf.x[6] += np.pi * 2

        new_theta = bbox[6]
        if new_theta >= np.pi: new_theta -= np.pi * 2    # make the theta still in the range
        if new_theta < -np.pi: new_theta += np.pi * 2
        bbox[6] = new_theta

        predicted_theta = self.kf.x[6]
        if abs(new_theta - predicted_theta) > np.pi / 2.0 and abs(new_theta - predicted_theta) < np.pi * 3 / 2.0:     # if the angle of two theta is not acute angle
            self.kf.x[6] += np.pi       
            if self.kf.x[6] > np.pi: self.kf.x[6] -= np.pi * 2    # make the theta still in the range
            if self.kf.x[6] < -np.pi: self.kf.x[6] += np.pi * 2

        # now the angle is acute: < 90 or > 270, convert the case of > 270 to < 90
        if abs(new_theta - self.kf.x[6]) >= np.pi * 3 / 2.0:
            if new_theta > 0: self.kf.x[6] += np.pi * 2
            else: self.kf.x[6] -= np.pi * 2

        #########################     # flip
        self.kf.update(bbox.reshape(7,1))

        if self.kf.x[6] >= np.pi: self.kf.x[6] -= np.pi * 2    # make the theta still in the rage
        if self.kf.x[6] < -np.pi: self.kf.x[6] += np.pi * 2

        bbox = self.kf.x.reshape(-1)[:9]
        self.bbox = bbox
        return self.bbox
