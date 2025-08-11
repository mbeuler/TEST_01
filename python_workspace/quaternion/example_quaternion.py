#
# Project: ZULS_0004
# Author : Marcel Beuler
# Date   : 2025-08-04
#

import numpy as np

from zulsquaternion import *

##########################################################################################
# Fundamental quaternion examples

print('Define two quaternions q1 and q2:')
q1 = zulsquaternion(1.0, 2.0, 3.0, 4.0)
q2 = zulsquaternion(5.0, 6.0, 7.0, 8.0)

print('q1 =', q1)
print('q2 =', q2)
print('')

print('Basic quaternion operations:')
q_add   = q1 + q2
q_sub   = q1 - q2
q_mul_1 = q1 * q2  # Multiplication of two quaternions
q_mul_2 = q1 * 0.5 # Multiplication of a quaternion and a scalar
q_mul_3 =  2 * q2  # Multiplication of a scalar and a quaternion
q1_norm = q1.norm()
q1_conj = q1.conjugate()
q1_normalize = q1.normalize()

print('q_add   =', q_add)
print('q_sub   =', q_sub)
print('q_mul_1 =', q_mul_1)
print('q_mul_2 =', q_mul_2)
print('q_mul_3 =', q_mul_3)
print('q1_norm =', q1_norm)
print('q1_conj =', q1_conj)
print('q1_normalize =', q1_normalize)
print('')

print('Conversion between euler angles and quaternion:')
phi   = 45.0 # DEG
theta = 30.0 # DEG
psi   = 60.0 # DEG

#phi   = 0.0 # DEG
#theta = 0.0 # DEG
#psi   = 0.0 # DEG --> leads to q = (1.0 + 0.0i + 0.0j + 0.0k)

print(f"Phi   = {phi  :.4f} deg")
print(f"Theta = {theta:.4f} deg")
print(f"Psi   = {psi  :.4f} deg")

phi   = np.radians(phi)   # Convert DEG into RAD
theta = np.radians(theta) # Convert DEG into RAD
psi   = np.radians(psi)   # Convert DEG into RAD

print('Function euler_to_quaternion()...')
q = zulsquaternion.euler_to_quaternion(phi, theta, psi)
print('q =', q)

print('Function quaternion_to_euler()...')
roll, pitch, yaw = q.quaternion_to_euler()

print(f"Phi   = {np.degrees(phi)  :.4f} deg")
print(f"Theta = {np.degrees(theta):.4f} deg")
print(f"Psi   = {np.degrees(psi)  :.4f} deg")
print('')
##########################################################################################

