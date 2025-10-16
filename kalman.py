# kalman.py
import numpy as np
import cv2

class KeypointKalman:
    """
    Per-keypoint simple Kalman smoother (2D position).
    Input: seq (T, K, 2) -> returns same shape smoothed.
    """

    def __init__(self, state_dim=4, meas_dim=2):
        self.state_dim = state_dim
        self.meas_dim = meas_dim

    def smooth_sequence(self, seq):
        T, K, C = seq.shape
        sm = np.zeros_like(seq, dtype=np.float32)
        for k in range(K):
            kf = cv2.KalmanFilter(self.state_dim, self.meas_dim)
            # state: [x, y, vx, vy]
            kf.transitionMatrix = np.array([[1,0,1,0],
                                            [0,1,0,1],
                                            [0,0,1,0],
                                            [0,0,0,1]], dtype=np.float32)
            kf.measurementMatrix = np.array([[1,0,0,0],
                                             [0,1,0,0]], dtype=np.float32)
            kf.processNoiseCov = np.eye(self.state_dim, dtype=np.float32) * 1e-4
            kf.measurementNoiseCov = np.eye(self.meas_dim, dtype=np.float32) * 1e-2
            # init
            first = seq[0, k, :2].astype(np.float32)
            kf.statePre = np.array([first[0], first[1], 0, 0], dtype=np.float32)
            kf.statePost = kf.statePre.copy()
            for t in range(T):
                meas = seq[t, k, :2].astype(np.float32)
                kf.predict()
                kf.correct(meas)
                s = kf.statePost
                sm[t, k, 0] = s[0]
                sm[t, k, 1] = s[1]
        return sm
