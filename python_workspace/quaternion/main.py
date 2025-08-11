#
# Project: ZULS_0004
# Author : Marcel Beuler
# Date   : 2025-08-05
#

import os
import numpy as np
import csv
import matplotlib.pyplot as plt

from read_csv import *
from zulsquaternion import *

##########################################################################################
directory = os.getcwd()        # Get path of the current working directory
os.chdir(directory + '\\Data') # Change path of the current working directory
                               # (folder 'Data' must already exist!)

# Constants
fS   = np.float32(1000.0);     # Sample frequency, [fS] = Hz
dt   = np.float32(1/fS)        # Sample time,      [dt] = s
fSSF = np.float32(65.5)        # Sensitivity scale factor according to IMU configuration
                               # [fSSF] = LSB/dps
fK1  = 1/(fS*fSSF)             # [fK1] = 1/(1/s * 1/(DEG/s)) = DEG
fK3  = np.float32(180.0/np.pi) # Convert radians into degrees
alpha= np.float32(0.9996)      # Complementary filter

# Read CSV data of corrected raw values in sensor coordinate system
Data = readCSVData()

# Data is a structured array whose datatype is a composition of simpler datatypes.
# That's why we cannot use the shape function.
if len(Data) == 0:
    exit(1)
elif len(Data.dtype.names) != 9:
    exit(1)

ui32SystemTime = Data[Data.dtype.names[0]] # First column
i16AccelX_CSV  = Data[Data.dtype.names[1]] # Second column
i16AccelY_CSV  = Data[Data.dtype.names[2]] # ...
i16AccelZ_CSV  = Data[Data.dtype.names[3]]
i16GyroX_CSV   = Data[Data.dtype.names[4]]
i16GyroY_CSV   = Data[Data.dtype.names[5]]
i16GyroZ_CSV   = Data[Data.dtype.names[6]]

fRollAngleNF_uC  = Data[Data.dtype.names[7]] # Roll angle (reference from uC)
fPitchAngleNF_uC = Data[Data.dtype.names[8]] # Pitch angle (reference from uC)

# Convert raw values in body fixed coordinate system
#i16AccelX_CSV = (+1)*i16AccelX_CSV # Not changed
i16AccelY_CSV = (-1)*i16AccelY_CSV
i16AccelZ_CSV = (-1)*i16AccelZ_CSV
#i16GyroX_CSV  = (+1)*i16GyroX_CSV  # Not changed
i16GyroY_CSV  = (-1)*i16GyroY_CSV
i16GyroZ_CSV  = (-1)*i16GyroZ_CSV

print('Length SystemTime =', len(ui32SystemTime))
print('')

# Declare arrays for all calculated roll and pitch angles
# - NF: No Filter (gyro drift occurs)
# - CF: Complementary Filter (gyro drift is compensated by accelerometer data)
fRollAngleNF_EULER  = np.zeros(len(ui32SystemTime), dtype=np.float32) # Angle in DEG!
fPitchAngleNF_EULER = np.zeros(len(ui32SystemTime), dtype=np.float32) # Angle in DEG!

fRollAngleCF_EULER  = np.zeros(len(ui32SystemTime), dtype=np.float32)  # Angle in DEG!
fPitchAngleCF_EULER = np.zeros(len(ui32SystemTime), dtype=np.float32)  # Angle in DEG!

fRollAngleNF_QUAT  = np.zeros(len(ui32SystemTime), dtype=np.float32)  # Angle in DEG!
fPitchAngleNF_QUAT = np.zeros(len(ui32SystemTime), dtype=np.float32)  # Angle in DEG!

fRollAngleCF_QUAT  = np.zeros(len(ui32SystemTime), dtype=np.float32)  # Angle in DEG!
fPitchAngleCF_QUAT = np.zeros(len(ui32SystemTime), dtype=np.float32)  # Angle in DEG!

print('Two integration methods for the quaternion differential equation are supported:')
print('1. Euler integration')
print('2. Runge-Kutta 4th order')
while True:
    keyboard = input('Please enter the number of the desired method (cancel with n): ')
    try:
        selector = int(keyboard)
        if selector == 1:
            integration_method = 'euler'
            print('You have selected the Euler integration method')
            break
        elif selector == 2:
            integration_method = 'rk4'
            print('You have selected the Runge-Kutta 4th order integration method')
            break
        else:
            print('Invalid method')
    except ValueError:
        if keyboard == 'N' or keyboard == 'n':
            exit(1)

##########################################################################################
print('')
print('Using differential equation of Euler angles...')

# Calculate fRollAngleNF_EULER and fPitchAngleNF_EULER ###################################
# Initialization                                                                        ##
fRollAngleNF  = np.float32(0.0)  # Angle in DEG!                                        ##
fPitchAngleNF = np.float32(0.0)  # Angle in DEG!                                        ##
                                                                                        ##
for i in range(len(ui32SystemTime)):                                                    ##
    # Read gyroscope values from CSV file                                               ##
    i16GyroX  = i16GyroX_CSV[i]                                                         ##
    i16GyroY  = i16GyroY_CSV[i]                                                         ##
    i16GyroZ  = i16GyroZ_CSV[i]                                                         ##
                                                                                        ##
    # Gyroscope angle calculation (body fixed system)                                   ##
    # Calculate the rotated roll angle and add it to the fRollAngleNF variable          ##
    # (no filter yet)                                                                   ##
    # Calculate the rotated pitch angle and add it to the fPitchangleNF variable        ##
    # (no filter yet)                                                                   ##
    fRollAngleNF_dot  = np.float32(i16GyroX) + np.tan(np.radians(fPitchAngleNF)) * \
                        (np.sin(np.radians(fRollAngleNF)) * np.float32(i16GyroY) + \
                         np.cos(np.radians(fRollAngleNF)) * np.float32(i16GyroZ))       ##
                                                                                        ##
    fPitchAngleNF_dot = np.cos(np.radians(fRollAngleNF)) * np.float32(i16GyroY) - \
                        np.sin(np.radians(fRollAngleNF)) * np.float32(i16GyroZ)         ##
                                                                                        ##
    fRollAngleNF  += fRollAngleNF_dot  * fK1  # Angle in DEG                            ##
    fPitchAngleNF += fPitchAngleNF_dot * fK1  # Angle in DEG                            ##
                                                                                        ##
    fRollAngleNF_EULER[i]  = fRollAngleNF                                               ##
    fPitchAngleNF_EULER[i] = fPitchAngleNF ###############################################
print('Calculation of fRollAngleNF_EULER and fPitchAngleNF_EULER finished')

# Calculate fRollAngleCF_EULER and fPitchAngleCF_EULER ###################################
# Initialization                                                                        ##
fRollAngleCF  = np.float32(0.0)  # Angle in DEG!                                        ##
fPitchAngleCF = np.float32(0.0)  # Angle in DEG!                                        ##
                                                                                        ##
for i in range(len(ui32SystemTime)):                                                    ##
    # Read accelerometer and gyroscope values from CSV file                             ##
    i16AccelX = i16AccelX_CSV[i]                                                        ##
    i16AccelY = i16AccelY_CSV[i]                                                        ##
    i16AccelZ = i16AccelZ_CSV[i]                                                        ##
    i16GyroX  = i16GyroX_CSV[i]                                                         ##
    i16GyroY  = i16GyroY_CSV[i]                                                         ##
    i16GyroZ  = i16GyroZ_CSV[i]                                                         ##
                                                                                        ##
    # Accelerometer angle calculation in DEG (body fixed system)                        ##
    i32Aux = np.int32(i16AccelY)*np.int32(i16AccelY) + \
             np.int32(i16AccelZ)*np.int32(i16AccelZ)                                    ##
                                                                                        ##
    fRollAngleAccel  = np.atan2(-i16AccelY, -i16AccelZ).astype(np.float32) * fK3        ##
    fPitchAngleAccel = np.atan2(i16AccelX, \
                            np.sqrt(i32Aux).astype(np.int16)).astype(np.float32) * fK3  ##
                                                                                        ##
    # Gyroscope angle calculation (body fixed system)                                   ##
    # Calculate the rotated roll angle and add it to the fRollAngleCF variable          ##
    # (complementary filter used)                                                       ##
    # Calculate the rotated pitch angle and add it to the fPitchangleCF variable        ##
    # (complementary filter used)                                                       ##
    fRollAngleCF_dot  = np.float32(i16GyroX) + np.tan(np.radians(fPitchAngleCF)) * \
                        (np.sin(np.radians(fRollAngleCF)) * np.float32(i16GyroY) + \
                         np.cos(np.radians(fRollAngleCF)) * np.float32(i16GyroZ))       ##
                                                                                        ##
    fPitchAngleCF_dot = np.cos(np.radians(fRollAngleCF)) * np.float32(i16GyroY) - \
                        np.sin(np.radians(fRollAngleCF)) * np.float32(i16GyroZ)         ##
                                                                                        ##
    fRollAngleCF  += fRollAngleCF_dot  * fK1  # Angle in DEG                            ##
    fPitchAngleCF += fPitchAngleCF_dot * fK1  # Angle in DEG                            ##
                                                                                        ##
    if i < 1:                                                                           ##
        # At 'i = 0' we get the first IMU value, i.e. the calculation has just started  ##
        # --> Set the CF roll angle equal to the accelerometer roll angle               ##
        # --> Set the CF pitch angle equal to the accelerometer pitch angle             ##
        fRollAngleCF  = fRollAngleAccel                                                 ##
        fPitchAngleCF = fPitchAngleAccel                                                ##
    else:                                                                               ##
        # Sensor fusion with complementary filter                                       ##
        fRollAngleCF  = fRollAngleCF  * alpha + (1.0 - alpha) * fRollAngleAccel         ##
        fPitchAngleCF = fPitchAngleCF * alpha + (1.0 - alpha) * fPitchAngleAccel        ##
                                                                                        ##
    fRollAngleCF_EULER[i]  = fRollAngleCF                                               ##
    fPitchAngleCF_EULER[i] = fPitchAngleCF ###############################################
print('Calculation of fRollAngleCF_EULER and fPitchAngleCF_EULER finished')

# Plot : fRollAngleNF_uC    & fPitchAngleNF_uC
#        fRollAngleNF_EULER & fPitchAngleNF_EULER
#        fRollAngleCF_EULER & fPitchAngleCF_EULER
SystemTime_plot = np.float32(ui32SystemTime / 1000)
fig, axs = plt.subplots(3, 1, figsize=(10, 6))

axs[0].plot(SystemTime_plot, fRollAngleNF_uC, color='b', \
            label='fRollAngleNF_uC')
axs[0].plot(SystemTime_plot, fPitchAngleNF_uC, color='r', \
            label='fPitchAngleNF_uC')
axs[0].set_ylabel('Angle [deg]')
axs[0].legend(loc='upper left',fontsize=8)
axs[0].grid(True)

axs[1].plot(SystemTime_plot, fRollAngleNF_EULER, color='b', \
            label='fRollAngleNF_EULER')
axs[1].plot(SystemTime_plot, fPitchAngleNF_EULER, color='r', \
            label='fPitchAngleNF_EULER')
axs[1].set_ylabel('Angle [deg]')
axs[1].legend(loc='upper left',fontsize=8)
axs[1].grid(True)

axs[2].plot(SystemTime_plot, fRollAngleCF_EULER, color='b', \
            label='fRollAngleCF_EULER')
axs[2].plot(SystemTime_plot, fPitchAngleCF_EULER, color='r', \
            label='fPitchAngleCF_EULER')
axs[2].set_ylabel('Angle [deg]')
axs[2].set_xlabel('Time [s]')
axs[2].legend(loc='upper left',fontsize=8)
axs[2].grid(True)

plt.tight_layout()
plt.savefig('example_euler.pdf', format='pdf')
#plt.show()

##########################################################################################
print('')
print('Using quaternion differential equation...')

# Calculate fRollAngleNF_QUAT and fPitchAngleNF_QUAT #####################################
# Initialization                                                                        ##
q = zulsquaternion(1.0, 0.0, 0.0, 0.0) # Initial quaternion (no rotation)               ##
                                                                                        ##
for i in range(len(ui32SystemTime)):                                                    ##
    # Read gyroscope values from CSV file                                               ##
    i16GyroX  = i16GyroX_CSV[i]                                                         ##
    i16GyroY  = i16GyroY_CSV[i]                                                         ##
    i16GyroZ  = i16GyroZ_CSV[i]                                                         ##
                                                                                        ##
    # Calculate omega (in RAD/s) from gyroscope data                                    ##
    omega_x = np.float32(i16GyroX) / fSSF  # [omega_x] = 1/[fSSF] = DEG/s               ##
    omega_y = np.float32(i16GyroY) / fSSF  # [omega_y] = 1/[fSSF] = DEG/s               ##
    omega_z = np.float32(i16GyroZ) / fSSF  # [omega_z] = 1/[fSSF] = DEG/s               ##
                                                                                        ##
    omega_x = omega_x * (np.pi / 180.0)  # Convert DEG/s into RAD/s                     ##
    omega_y = omega_y * (np.pi / 180.0)  # Convert DEG/s into RAD/s                     ##
    omega_z = omega_z * (np.pi / 180.0)  # Convert DEG/s into RAD/s                     ##
                                                                                        ##
    omega = (omega_x, omega_y, omega_z)                                                 ##
                                                                                        ##
    if integration_method == 'euler':                                                   ##
        q = q.integrate_euler(q, omega, dt)  # Euler integration                        ##
    elif integration_method == 'rk4':                                                   ##
        q = q.integrate_rk4(q, omega, dt)    # Runge-Kutta 4th order integration        ##
    else:                                                                               ##
        exit(1)                                                                         ##
                                                                                        ##
    roll, pitch, yaw = q.quaternion_to_euler()      # RAD                               ##
                                                                                        ##
    fRollAngleNF_QUAT[i]  = roll  * (180.0 / np.pi) # Convert RAD into DEG              ##
    fPitchAngleNF_QUAT[i] = pitch * (180.0 / np.pi) # Convert RAD into DEG ###############
print('Calculation of fRollAngleNF_QUAT and fPitchAngleNF_QUAT finished')

# Calculate fRollAngleCF_QUAT and fPitchAngleCF_QUAT #####################################
# Initialization                                                                        ##
q = zulsquaternion(1.0, 0.0, 0.0, 0.0) # Initial quaternion (no rotation)               ##
                                                                                        ##
for i in range(len(ui32SystemTime)):                                                    ##
    # Read accelerometer and gyroscope values from CSV file                             ##
    i16AccelX = i16AccelX_CSV[i]                                                        ##
    i16AccelY = i16AccelY_CSV[i]                                                        ##
    i16AccelZ = i16AccelZ_CSV[i]                                                        ##
    i16GyroX  = i16GyroX_CSV[i]                                                         ##
    i16GyroY  = i16GyroY_CSV[i]                                                         ##
    i16GyroZ  = i16GyroZ_CSV[i]                                                         ##
                                                                                        ##
    # Accelerometer angle calculation in RAD (body fixed system)                        ##
    i32Aux = np.int32(i16AccelY)*np.int32(i16AccelY) + \
             np.int32(i16AccelZ)*np.int32(i16AccelZ)                                    ##
                                                                                        ##
    fRollAngleAccel  = np.atan2(-i16AccelY, -i16AccelZ).astype(np.float32)              ##
    fPitchAngleAccel = np.atan2(i16AccelX, \
                            np.sqrt(i32Aux).astype(np.int16)).astype(np.float32)        ##
                                                                                        ##
    # Convert accelerometer Euler angles (in RAD!) into quaternion                      ##
    q_accel = zulsquaternion.euler_to_quaternion(fRollAngleAccel,fPitchAngleAccel,0)    ##
                                                                                        ##
    # Calculate omega (in RAD/s) from gyroscope data                                    ##
    omega_x = np.float32(i16GyroX) / fSSF  # [omega_x] = 1/[fSSF] = DEG/s               ##
    omega_y = np.float32(i16GyroY) / fSSF  # [omega_y] = 1/[fSSF] = DEG/s               ##
    omega_z = np.float32(i16GyroZ) / fSSF  # [omega_z] = 1/[fSSF] = DEG/s               ##
                                                                                        ##
    omega_x = omega_x * (np.pi / 180.0)  # Convert DEG/s into RAD/s                     ##
    omega_y = omega_y * (np.pi / 180.0)  # Convert DEG/s into RAD/s                     ##
    omega_z = omega_z * (np.pi / 180.0)  # Convert DEG/s into RAD/s                     ##
                                                                                        ##
    omega = (omega_x, omega_y, omega_z)                                                 ##
                                                                                        ##
    if integration_method == 'euler':                                                   ##
        q = q.integrate_euler(q, omega, dt)  # Euler integration                        ##
    elif integration_method == 'rk4':                                                   ##
        q = q.integrate_rk4(q, omega, dt)    # Runge-Kutta 4th order integration        ##
    else:                                                                               ##
        exit(1)                                                                         ##
                                                                                        ##
    if i < 1:                                                                           ##
        # At 'i = 0' we get the first IMU value, i.e. the calculation has just started  ##
        # --> Set q to q_accel                                                          ##
        q = q_accel                                                                     ##
    else:                                                                               ##
        # Sensor fusion with complementary filter                                       ##
        q = alpha * q + (1 - alpha) * q_accel                                           ##
        q = q.normalize()                                                               ##
                                                                                        ##
    roll, pitch, yaw = q.quaternion_to_euler()      # RAD                               ##
                                                                                        ##
    fRollAngleCF_QUAT[i]  = roll  * (180.0 / np.pi) # Convert RAD into DEG              ##
    fPitchAngleCF_QUAT[i] = pitch * (180.0 / np.pi) # Convert RAD into DEG ###############
print('Calculation of fRollAngleCF_QUAT and fPitchAngleCF_QUAT finished')    

# Plot : fRollAngleNF_uC   & fPitchAngleNF_uC
#        fRollAngleNF_QUAT & fPitchAngleNF_QUAT
#        fRollAngleCF_QUAT & fPitchAngleCF_QUAT
fig, axs = plt.subplots(3, 1, figsize=(10, 6))

axs[0].plot(SystemTime_plot, fRollAngleNF_uC, color='b', \
            label='fRollAngleNF_uC')
axs[0].plot(SystemTime_plot, fPitchAngleNF_uC, color='r', \
            label='fPitchAngleNF_uC')
axs[0].set_ylabel('Angle [deg]')
axs[0].legend(loc='upper left',fontsize=8)
axs[0].grid(True)

axs[1].plot(SystemTime_plot, fRollAngleNF_QUAT, color='b', \
            label='fRollAngleNF_QUAT')
axs[1].plot(SystemTime_plot, fPitchAngleNF_QUAT, color='r', \
            label='fPitchAngleNF_QUAT')
axs[1].set_ylabel('Angle [deg]')
axs[1].legend(loc='upper left',fontsize=8)
axs[1].grid(True)

axs[2].plot(SystemTime_plot, fRollAngleCF_QUAT, color='b', \
            label='fRollAngleCF_QUAT')
axs[2].plot(SystemTime_plot, fPitchAngleCF_QUAT, color='r', \
            label='fPitchAngleCF_QUAT')
axs[2].set_ylabel('Angle [deg]')
axs[2].set_xlabel('Time [s]')
axs[2].legend(loc='upper left',fontsize=8)
axs[2].grid(True)

plt.tight_layout()
plt.savefig('example_quat.pdf', format='pdf')
plt.show()
##########################################################################################

#print('Last roll, pitch and yaw angle:')
#print(f"Roll : {roll  * (180.0 / np.pi):.2f} deg")
#print(f"Pitch: {pitch * (180.0 / np.pi):.2f} deg")
#print(f"Yaw  : {yaw   * (180.0 / np.pi):.2f} deg")

