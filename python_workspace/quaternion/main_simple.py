#
# Project: ZULS_0004
# Author : Marcel Beuler
# Date   : 2025-07-30
#

import numpy as np
import matplotlib.pyplot as plt

from zulsquaternion import *

##########################################################################################
# Constants
dt = 0.001  # [dt] = s

# Initialization
q = zulsquaternion(1.0, 0.0, 0.0, 0.0) # Initial quaternion (no rotation)

array_length = 9000
SystemTime = np.arange(1, array_length+1, dtype=np.int32)

fGyroX_deg = np.zeros(len(SystemTime), dtype=np.float32)
fGyroY_deg = np.zeros(len(SystemTime), dtype=np.float32)
fGyroZ_deg = np.zeros(len(SystemTime), dtype=np.float32)

fGyroY_deg[:3000] = 15.0   # Define pitch rate in DEG/s
fGyroZ_deg[3000:] = -15.0  # Define yaw rate in DEG/s

fRollAngle  = np.zeros(len(SystemTime), dtype=np.float32)
fPitchAngle = np.zeros(len(SystemTime), dtype=np.float32)

for i in range(len(SystemTime)):
    omega_x = fGyroX_deg[i]
    omega_y = fGyroY_deg[i]
    omega_z = fGyroZ_deg[i]

    omega_x = omega_x * (np.pi / 180.0)  # Convert DEG/s into RAD/s
    omega_y = omega_y * (np.pi / 180.0)  # Convert DEG/s into RAD/s
    omega_z = omega_z * (np.pi / 180.0)  # Convert DEG/s into RAD/s

    omega = (omega_x, omega_y, omega_z)

    q = q.integrate(omega, dt)

    roll, pitch, yaw = q.quaternion_to_euler()      # RAD
    fRollAngle[i]  = roll  * (180.0 / np.pi) # Convert RAD into DEG
    fPitchAngle[i] = pitch * (180.0 / np.pi) # Convert RAD into DEG

# Print last roll, pitch and yaw angle
print(f"Roll : {roll  * (180.0 / np.pi):.2f} deg")
print(f"Pitch: {pitch * (180.0 / np.pi):.2f} deg")
print(f"Yaw  : {yaw   * (180.0 / np.pi):.2f} deg")

plt.figure(figsize=(10, 5))
plt.plot(SystemTime/1000, fRollAngle, label='fRollAngle')
plt.plot(SystemTime/1000, fPitchAngle, label='fPitchAngle')
plt.ylabel('Angle [deg]')
plt.xlabel('Time [s]')
plt.legend(loc='lower left')
plt.grid()
plt.show()
##########################################################################################

