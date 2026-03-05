# Robot Control

Solutions for the Robot Control course at the University of Warsaw, 2025/2026.

Course page [here](https://mim-ml-teaching.github.io/public-rc-2025-26/).

## Assignments

| # | Topic | What's inside |
|---|-------|---------------|
| [rc1](rc1/) | Image stitching | Camera calibration (ChArUco), image undistortion, projective transformations, ORB + RANSAC feature matching |
| [rc2](rc2/) | Drone PID control | PID controller design, Skydio X2 drone simulation in MuJoCo, flying through gates with wind and rotation |
| [rc3](rc3/) | Drone vision & Kalman filter | ArUco-based pose estimation, Kalman filter for smoothing, vision-based gate navigation |

Each folder has its own README / report with details.

## Quick start

All assignments use [uv](https://github.com/astral-sh/uv) for dependency management:

```bash
cd rc3
uv sync
```
