import numpy as np


class KalmanFilter:
    def __init__(
        self, 
        delta_t: float,
        R: np.ndarray | None = None,
        sigma_q: float = 0.1
    ):
        self.F = np.eye(6, dtype=float) # 1 for position
        for i in range(3):
            # delta t for velocity
            self.F[i, i + 3] = delta_t 

        self.H = np.zeros((3, 6), dtype=float)
        for i in range(3):
            # only measurements for position
            self.H[i, i] = 1

        self.R = R if R is not None else np.array([
            [ 0.0000363538,  0.0000005538,  0.0000164871],
            [ 0.0000005538,  0.0000024301, -0.0000002239],
            [ 0.0000164871, -0.0000002239,  0.0000106944]
        ]) # measurement I got from experiments
        
        q_vals = sigma_q * np.array([
            [delta_t ** 4 / 4, delta_t ** 3 / 2],
            [delta_t ** 3 / 2, delta_t ** 2]
        ])
        self.Q = np.zeros((6, 6), dtype=float)
        for i in range(3):
            self.Q[i, i] = q_vals[0, 0]
            self.Q[i, i + 3] = q_vals[0, 1]
            self.Q[i + 3, i] = q_vals[1, 0]
            self.Q[i + 3, i + 3] = q_vals[1, 1]
        
        self.P_last = 100 * np.eye(6, dtype=float)
        self.x_last = np.zeros(6, dtype=float)

    def predict(self):
        self.x_pred = self.F @ self.x_last
        self.P_pred = self.F @ self.P_last @ self.F.T + self.Q
        return self.x_pred[:3] # return only positions

    def update(self, z: np.ndarray) -> np.ndarray:
        y = z - self.H @ self.x_pred
        K = self.P_pred @ self.H.T @ np.linalg.inv(self.H @ self.P_pred @ self.H.T + self.R)
        
        x_new = self.x_pred + K @ y
        p_new = (np.eye(6) - K @ self.H) @ self.P_pred
        
        self.x_last = x_new
        self.P_last = p_new

        return x_new[:3] # return only positions

    def step_blind(self):
        self.predict()
        self.x_last = self.x_pred
        self.P_last = self.P_pred
        return self.x_pred[:3]