import numpy as np
import pandas as pd
import cv2
import math
import tyro
import mujoco
from typing import Optional
from mujoco import viewer
from drone_simulator import DroneSimulator
from pid import PID
from plotting import plot_results, plot_orientation_results
# TODO: Additional imports if needed
from typing import Iterable
import kalman_filter as kf
# END OF TODO

# Simulation parameters
resolution = (480, 640)  # (height, width) in pixels
fovy_deg = 90  # vertical field of view in degrees
np.set_printoptions(suppress=True, precision=10)
pd.set_option('display.float_format', lambda x: f'{x:.3f}')

# TODO: Additional functions if needed
# Camera has some offset from the center of the drone specified in x2.xml.
CAMERA_OFFSET = np.array([-0.16, 0.0, 0.02])

# Opencv and mujoco use different coordinate systems.
# This matrix converts a column of mujoco coordinates to cv coordinates.
# X_cv = MUJOCO_TO_CV @ X_muj
MUJOCO_TO_CV = np.array([
    [0, 1, 0],
    [0, 0, -1],
    [-1, 0, 0]
])
CV_TO_MUJOCO = np.linalg.inv(MUJOCO_TO_CV)

ARUCO_SIZE = 0.1
ARUCO_MIDDLES = np.array((
    (-0.01, -0.6, 0.65),      # 4 * n
    (-0.01, 0.6, 0.65),     # 4 * n + 1
    (-0.01, 0.6, -0.65),    # 4 * n + 2
    (-0.01, -0.6, -0.65)      # 4 * n + 3
))
ARUCO_CORNER_TRANSLATIONS = np.array((
    (-ARUCO_SIZE, -ARUCO_SIZE, ARUCO_SIZE),      # top right
    (-ARUCO_SIZE, -ARUCO_SIZE, -ARUCO_SIZE),     # bottom right
    (-ARUCO_SIZE, ARUCO_SIZE, -ARUCO_SIZE),    # bottom left
    (-ARUCO_SIZE, ARUCO_SIZE, ARUCO_SIZE),     # top left
))

ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
DETECTOR_PARAMS = cv2.aruco.DetectorParameters()
aruco_detector = cv2.aruco.ArucoDetector(
    ARUCO_DICT,
    DETECTOR_PARAMS
)

# Opencv aruco marker detection and pose estimation functions

def find_markers(
    camera_frame: np.ndarray, 
    id_range: Iterable[int]
) -> tuple[np.ndarray, np.ndarray]:
    camera_frme_bgr = cv2.cvtColor(camera_frame, cv2.COLOR_RGB2BGR)

    found_corners, found_ids, _ = aruco_detector.detectMarkers(camera_frme_bgr)
    if len(found_corners) == 0:
        return np.array([]), np.array([])

    found_corners = np.array(found_corners)
    found_ids = found_ids.reshape(-1)
    
    result_corners = []
    used_ids = []
    result_ids = []

    for i in id_range:
        idx = np.where(found_ids == i)
        if len(idx[0]) > 0:
            result_corners.append(found_corners[idx[0]])
            used_ids.append(True)
            result_ids.append(i)
        else:
            used_ids.append(False)
    
    result_corners = np.array(result_corners)
    used_ids = np.array(used_ids)
    return result_corners, used_ids

def sovlve_pnp(
    corners_2d: np.ndarray,
    used_corner_ids: np.ndarray,
    camera_matrix: np.ndarray, 
    dist_coeffs: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:

    corners_3d = []
    for i, middle in enumerate(ARUCO_MIDDLES):
        if used_corner_ids[i] == False:
            continue
        for translation in ARUCO_CORNER_TRANSLATIONS:
            corners_3d.append(middle + translation)
    corners_3d = np.array(corners_3d)
    corners_3d = (MUJOCO_TO_CV @ corners_3d.T).T

    corners_2d = corners_2d.reshape(-1, 2)
    retval, rvec, tvec = cv2.solvePnP(
        corners_3d,
        corners_2d,
        camera_matrix,
        dist_coeffs,
    )
    assert retval == True
    tvec = (CV_TO_MUJOCO @ tvec.reshape(-1).T).T
    return rvec, tvec


# Some angle magic

def get_roll_pitch_yaw(rvec):
    R_cv = cv2.Rodrigues(rvec)[0]
    R_muj = CV_TO_MUJOCO @ R_cv @ MUJOCO_TO_CV
    
    pitch = math.atan2(-R_muj[0, 2], math.sqrt(R_muj[1, 2]**2 + R_muj[2, 2]**2))
    yaw = math.atan2(-R_muj[0, 1], -R_muj[0, 0])
    roll = math.atan2(R_muj[1, 2], R_muj[2, 2])

    return np.array([roll, pitch, yaw])

def get_true_relative_roll_pitch_yaw(model, data, gate_name):
    drone_id = model.body("x2").id 
    drone_mat = data.xmat[drone_id].reshape(3, 3)

    gate_id = model.body(gate_name).id
    gate_mat = data.xmat[gate_id].reshape(3, 3)

    R_rel = gate_mat.T @ drone_mat

    pitch = math.atan2(-R_rel[2, 0], math.sqrt(R_rel[2, 1]**2 + R_rel[2, 2]**2))
    yaw = math.atan2(R_rel[1, 0], R_rel[0, 0])
    roll = math.atan2(R_rel[2, 1], R_rel[2, 2])

    return np.array([roll, pitch, yaw])
# END OF TODO


def camera_intrinsics_from_fovy(fovy_deg: float, height: int, width: int) -> np.ndarray:
    fy = (height / 2.0) / math.tan(math.radians(fovy_deg) / 2.0)
    K = np.array([[fy, 0,  width / 2.0],
                  [0,  fy, height / 2.0],
                  [0,   0,  1]])
    return K


def update_gate_position(model: mujoco.MjModel, data: mujoco.MjData, gate_name: str, gate_vel: np.ndarray, 
                         gate_motion_prob: float = 0.1,
                         gate_motion_scale: float = 0.1, 
                         gate_noise_scale: float = 0.005, 
                         gate_damping: float = 0.95,
                         sim_dt: Optional[float] = None) -> np.ndarray:
    """Update gate position with smooth motion dynamics."""
    if sim_dt is None:
        sim_dt = model.opt.timestep
    gate_id = model.body(gate_name).id

    # Random impulse
    if np.random.uniform() < gate_motion_prob:
        impulse = np.random.normal(scale=gate_motion_scale, size=3)
        gate_vel += impulse
    
    # Add noise and apply damping
    gate_vel += np.random.normal(scale=gate_noise_scale, size=3)
    gate_vel *= gate_damping
    
    # Integrate to new position
    cur_pos = data.body(gate_name).xpos.copy()
    new_pos = cur_pos + gate_vel * sim_dt * [0.3, 1.0, 1.0]  # slower x motion
    
    # Keep gate within reasonable bounds
    new_pos[0] = np.clip(new_pos[0], -10.0, 2.0)  # x bounds
    new_pos[1] = np.clip(new_pos[1], -1.5, 1.5)   # y bounds
    new_pos[2] = np.clip(new_pos[2], 0.5, 5)      # z bounds
    
    # Apply to model
    model.body_pos[gate_id] = new_pos
    
    return gate_vel


def build_world(rotated_gates: bool) -> str:
    world = open("scene.xml").read()


        # Use random starting positions
    world = world.replace(
        '<body name="red_gate" pos="-2 0 3">',
        f'<body name="red_gate" pos="-2 {np.random.uniform(-0.6, 0.6)} {np.random.uniform(2.7, 3.3)}">'
    )
    
    world = world.replace(
        '<body name="green_gate" pos="-4 -0.6 3.3">',
        f'<body name="green_gate" pos="-4 {np.random.uniform(-0.6, 0.6)} {np.random.uniform(2.7, 3.3)}">'
    )
    world = world.replace(
        '<body name="blue_gate" pos="-6 0.6 2.7">',
        f'<body name="blue_gate" pos="-6 {np.random.uniform(-0.6, 0.6)} {np.random.uniform(2.7, 3.3)}">'
    )

    if rotated_gates:
        world = world.replace(
            '<body name="red_gate"',
            f'<body name="red_gate" euler="0 0 {np.random.uniform(-30, -15) if np.random.rand() < 0.5 else np.random.uniform(15, 30)}"'
        )
        world = world.replace(
            '<body name="green_gate"',
            f'<body name="green_gate" euler="0 0 {np.random.uniform(-30, -15) if np.random.rand() < 0.5 else np.random.uniform(15, 30)}"'
        )
        world = world.replace(
            '<body name="blue_gate"',
            f'<body name="blue_gate" euler="0 0 {np.random.uniform(-30, -15) if np.random.rand() < 0.5 else np.random.uniform(15, 30)}"'
        )
    return world


def run_single_task(*, wind: bool, rotated_gates: bool, flight_mode, rendering_freq: float) -> None:
    world = build_world(rotated_gates)
    model = mujoco.MjModel.from_xml_string(world)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    view = viewer.launch_passive(model, data)
    view.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
    view.cam.fixedcamid = model.camera("front_camera").id

    K = camera_intrinsics_from_fovy(fovy_deg, resolution[0], resolution[1])
    dist_coeffs = np.zeros(5)
    desired_thrust = 3.2496
    roll_thrust, pitch_thrust, yaw_thrust = 0.0, 0.0, 0.0

    SIM_TIME = 500 if flight_mode == "hover" else 5000

    pnp_position_list = []  # Store PnP position estimates
    true_position_list = []  # Store true gate positions
    kalman_position_list = []  # Store KF filtered positions
    pnp_orientation_list = []
    true_orientation_list = []
    kalman_orientation_list = []
    # TODO: Additional variables if needed
    rvecs_list = []

    pnp_position = np.zeros(3)
    roll_pitch_yaw = np.zeros(3)
    kalman_position = np.zeros(3)
    current_marker = 0

    kalman_filter_pos = kf.KalmanFilter(model.opt.timestep)
    kalman_filter_ang = kf.KalmanFilter(model.opt.timestep)
    # PIDs
    pid_x = PID(
        gain_prop = 2, gain_int = 0.01, gain_der = 1,
        sensor_period = model.opt.timestep, output_limits=(-0.05, 0.05)
    )
    pid_y = PID(
        gain_prop = 2, gain_int = 0.001, gain_der = 1,
        sensor_period = model.opt.timestep, output_limits=(-0.05, 0.05)
    )

    pid_roll = PID(
        gain_prop = -0.5, gain_int = -0.001, gain_der = -0.05,
        sensor_period = model.opt.timestep, output_limits=(-5, 5)
    )
    pid_pitch = PID(
        gain_prop = -0.5, gain_int = -0.001, gain_der = -0.05,
        sensor_period = model.opt.timestep, output_limits=(-1, 1)
    )
    pid_yaw = PID(
        gain_prop = 2, gain_int = 0.001, gain_der = 5,
        sensor_period = model.opt.timestep, output_limits=(-1, 1)
    )
    pid_thrust = PID(
        gain_prop = 5, gain_int = 0.15, gain_der = 2,
        sensor_period = model.opt.timestep, output_limits=(-1, 1)
    )

    pids = [
        pid_roll,
        pid_pitch,
        pid_yaw,
        pid_thrust,
        pid_x,
        pid_y
    ]

    phase = 0
    previous_pos = np.zeros(3)
    prev_roll_pitch_yaw = np.zeros(3)
    # END OF TODO

    task_label = f"rotated={'yes' if rotated_gates else 'no'}, wind={'yes' if wind else 'no'}, flight_mode={flight_mode}"
    print(f"Starting task ({task_label})")

    wind_change_prob = 0.1 if wind else 0

    # If you want the simulation to be displayed more slowly, decrease rendering_freq
    # Note that this DOES NOT change the timestep used to approximate the physics of the simulation!
    drone_simulator = DroneSimulator(
        model, data, view, wind_change_prob = wind_change_prob, rendering_freq = rendering_freq
    )

    # --- initiate motion for all 3 gates ---
    red_gate_vel = np.zeros(3, dtype=float)
    green_gate_vel = np.zeros(3, dtype=float)
    blue_gate_vel = np.zeros(3, dtype=float)
    # ---------------------------------------

    renderer = None
    try:
        renderer = mujoco.Renderer(model, resolution[0], resolution[1])
        for i in range(SIM_TIME):
            # ----- update smooth motion of all 3 gates -----
            # Move the gates only in "hover" mode
            if flight_mode == "hover":
                red_gate_vel = update_gate_position(model, data, "red_gate", red_gate_vel)
                green_gate_vel = update_gate_position(model, data, "green_gate", green_gate_vel)
                blue_gate_vel = update_gate_position(model, data, "blue_gate", blue_gate_vel)
            mujoco.mj_forward(model, data)
            # -----------------------------------------------
            
            # Render camera frame
            renderer.update_scene(data, camera="front_camera")
            camera_frame = renderer.render()
            camera_frame = np.asarray(camera_frame, dtype=np.uint8)

            # Get current orientation
            # current_orien, _ = drone_simulator.orientation_sensor()

            drone_position = drone_simulator.position_sensor()[0]
            if current_marker == 0:
                gate_position = data.body("red_gate").xpos.copy()
            elif current_marker == 4:
                gate_position = data.body("green_gate").xpos.copy()
            elif current_marker == 8:
                gate_position = data.body("blue_gate").xpos.copy()

            # You can use true_position as ground truth for debugging purposes
            true_position = gate_position - drone_position
            # print(f"true_position: {true_position.round(3)}")

            # TODO: Detect, estimate pose, apply Kalman filter
            corners, used_ids = find_markers(
                camera_frame, 
                range(current_marker, current_marker+4)
            )

            if len(corners) == 0:
                kalman_position = kalman_filter_pos.step_blind()
                kalman_roll_pitch_yaw = kalman_filter_ang.step_blind()
            else:
                rvec, tvec = sovlve_pnp(corners, used_ids, K, dist_coeffs)
                rvecs_list.append(rvec.reshape(-1))

                roll_pitch_yaw = get_roll_pitch_yaw(rvec)
                pnp_position = tvec + CAMERA_OFFSET
                
                kalman_position = kalman_filter_pos.predict()
                kalman_roll_pitch_yaw = kalman_filter_ang.predict()

                kalman_position = kalman_filter_pos.update(pnp_position)
                kalman_roll_pitch_yaw = kalman_filter_ang.update(roll_pitch_yaw)

                current_pos = kalman_position.copy()
                current_roll_pitch_yaw = kalman_roll_pitch_yaw

            current_pos = kalman_position
            current_roll_pitch_yaw = kalman_roll_pitch_yaw

            if current_marker == 0:
                target_gate = "red_gate"
            elif current_marker == 4:
                target_gate = "green_gate"
            elif current_marker == 8:
                target_gate = "blue_gate"
            true_ref_orientation = get_true_relative_roll_pitch_yaw(model, data, target_gate)

            pnp_orientation_list.append(roll_pitch_yaw)
            kalman_orientation_list.append(kalman_roll_pitch_yaw)
            true_orientation_list.append(true_ref_orientation)
            
            if flight_mode == "flight":
                current_pos -= CAMERA_OFFSET

                yaw_rad = current_roll_pitch_yaw[2]
                translation_vec = np.array([0.5 * math.cos(yaw_rad), 0.5 * math.sin(yaw_rad), 0.0])
                if phase == 0:
                    pos_target = -translation_vec
                else:
                    pos_target = np.zeros(3)
                
                pos_target[2] -= 0.2


                desired_yaw = 180 + math.degrees(math.atan2(
                    current_pos[1], 
                    current_pos[0]
                ))
                yaw_error = desired_yaw - current_roll_pitch_yaw[2]
                yaw_error = (yaw_error + 180) % 360 - 180
                desired_yaw = current_roll_pitch_yaw[2] + yaw_error
                
                desired_x = pid_x.output_signal(pos_target[0], [current_pos[0], previous_pos[0]])
                desired_y = pid_y.output_signal(pos_target[1], [current_pos[1], previous_pos[1]])
            
                desired_pitch = 0.0-desired_x
                desired_roll = 0.0-desired_y

                pitch_thrust = pid_pitch.output_signal(desired_pitch, [current_roll_pitch_yaw[1], prev_roll_pitch_yaw[1]])
                roll_thrust = pid_roll.output_signal(-desired_roll, [current_roll_pitch_yaw[0], prev_roll_pitch_yaw[0]])
                yaw_thrust = pid_yaw.output_signal(desired_yaw, [current_roll_pitch_yaw[2], prev_roll_pitch_yaw[2]])

                desired_thrust = 3.2496 + pid_thrust.output_signal(pos_target[2], [-current_pos[2], -previous_pos[2]])
                
                previous_pos = current_pos
                prev_roll_pitch_yaw = current_roll_pitch_yaw
                
                dist = np.linalg.norm(current_pos - pos_target)
                print(len(corners), "markers detected, step:", i)
                print("   distance to target:", dist)
                if current_marker != 8 and dist < 0.4:
                    for pid in pids:
                        pid.err_sum = 0
                    if phase == 1:
                        current_marker += 4
                        kalman_filter_pos = kf.KalmanFilter(model.opt.timestep)
                        kalman_filter_ang = kf.KalmanFilter(model.opt.timestep)
                        phase = 0
                    else:
                        phase = 1
                print("current_marker:", current_marker, "phase:", phase)
                print("   desired_roll: {:.3f}, desired_pitch: {:.3f}, desired_yaw: {:.3f}, desired_thrust: {:.3f}".format(
                    desired_roll, desired_pitch, desired_yaw, desired_thrust
                ))
                print("   current_roll: {:.3f}, current_pitch: {:.3f}, current_yaw: {:.3f}".format(
                    current_roll_pitch_yaw[0], current_roll_pitch_yaw[1], current_roll_pitch_yaw[2]
                ))

            # END OF TODO

            if pnp_position is not None:
                pnp_position_list.append(pnp_position.copy())
                kalman_position_list.append(kalman_position.copy())
                true_position_list.append(true_position.copy())


            # Make a simulation step
            drone_simulator.sim_step(
                desired_thrust, roll_thrust,
                pitch_thrust, yaw_thrust
            )

        # Plot the results
        rotated_str = "rotated" if rotated_gates else "straight"
        mode_str = "flight" if flight_mode == "flight" else "hover"
        filename = f"plot_{rotated_str}_{mode_str}.png"
        plot_results(pnp_position_list, true_position_list, kalman_position_list, filename=filename)

        orientation_filename = f"plot_orientation_{rotated_str}_{mode_str}.png"
        plot_orientation_results(
            pnp_orientation_list,
            true_orientation_list,
            kalman_orientation_list,
            filename=orientation_filename
        )

        pnp_np = np.array(pnp_position_list).T
        R_matrix = np.cov(pnp_np - np.array(true_position_list).T, bias=False)
        print("Covariance matrix of PnP position estimates:")
        print(R_matrix)
        print("RMSE of PnP position estimates:")
        rmse = np.sqrt(np.mean((pnp_np - np.array(true_position_list).T) ** 2, axis=1))
        print(rmse)
        print("RMSE of Kalman Filter position estimates:")
        kalman_np = np.array(kalman_position_list).T
        rmse_kalman = np.sqrt(np.mean((kalman_np - np.array(true_position_list).T) ** 2, axis=1))
        print(rmse_kalman)
        print(f"Task ({task_label}) completed successfully!")

    finally:
        # Ensure renderer and viewer are closed before the next run to avoid multiple open windows.
        if renderer is not None:
            try:
                renderer.close()
            except Exception:
                pass
        try:
            view.close()
        except Exception:
            pass
    


def main(
    wind: bool = False,
    rotated_gates: bool = False,
    all_tasks: bool = True,
    runs: int = 1,
    rendering_freq: float = 1
) -> None:
    """
    Run the drone control simulation.

    Args:
        wind: Enable wind disturbances.
        rotated_gates: Rotate gates to create the harder variant.
        all_tasks: Run all four combinations of wind/rotated gates.
        runs: How many times to repeat each selected task.
        rendering_freq: Viewer rendering frequency multiplier (lower slows playback).
    """
    task_list = []
    if all_tasks:
        task_list = [
            (False, False, "flight"),
            (False, True, "hover"),
        ]
    else:
        task_list = [(False, False, "hover")] # the easiest setup

    for wind_flag, rotated, flight_mode in task_list:
        for run_idx in range(runs):
            print(f"\nRun {run_idx + 1}/{runs} for wind={wind_flag}, rotated_gates={rotated}, flight_mode={flight_mode}")
            run_single_task(
                wind=wind_flag,
                rotated_gates=rotated,
                flight_mode=flight_mode,
                rendering_freq=rendering_freq
            )


if __name__ == '__main__':
    tyro.cli(main)
