#!/usr/bin/env python3
from dataset import get_gaze_pose
import numpy as np
"""
mpiigaze range:gaze pitch(-21, 1) yaw(-20, 20)  pose pitch(-28, 42), yaw(-25, 35)
"""

print(get_gaze_pose())
