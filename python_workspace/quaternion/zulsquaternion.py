#
# Project: ZULS_0004
# Author : Marcel Beuler
# Date   : 2025-08-04
#

import numpy as np
import math

##########################################################################################
class zulsquaternion:
    def __init__(self, w, x, y, z):
        self.elements = np.array([w, x, y, z], dtype=np.float32)

    @property
    def w(self):
        return self.elements[0]

    @property
    def x(self):
        return self.elements[1]

    @property
    def y(self):
        return self.elements[2]

    @property
    def z(self):
        return self.elements[3]
##########################################################################################


##########################################################################################
    def __add__(self, other):
        if isinstance(other, zulsquaternion):
            return zulsquaternion(*(self.elements + other.elements))
        else:
            raise TypeError('Addition only supports another Quaternion.')
##########################################################################################


##########################################################################################
    def __sub__(self, other):
        if isinstance(other, zulsquaternion):
            return zulsquaternion(*(self.elements - other.elements))
        else:
            raise TypeError('Subtraction only supports another Quaternion.')
##########################################################################################


##########################################################################################
    def __mul__(self, other):
        if isinstance(other, zulsquaternion):
            # Quaternion multiplication
            w1, x1, y1, z1 = self.elements
            w2, x2, y2, z2 = other.elements
            return zulsquaternion(
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
            )
        elif isinstance(other, (int, float, np.float32)):
            # Ensure that the result is also np.float32
            return zulsquaternion(*(self.elements * np.float32(other)))
        else:
            raise TypeError("Multiplication supports another Quaternion or a scalar.")


    def __rmul__(self, other):
        if isinstance(other, (int, float, np.float32)):
            return zulsquaternion(*(self.elements * np.float32(other)))
        else:
            raise TypeError("Multiplication supports a scalar.")
##########################################################################################


##########################################################################################
    def norm(self):
        return np.sqrt(np.sum(self.elements ** 2))
        #return np.sqrt(self.w**2 + self.x**2 + self.y**2 + self.z**2)
##########################################################################################


##########################################################################################
    def normalize(self):
        norm = self.norm()
        if norm == 0:
            raise ValueError("Cannot normalize a zero-length quaternion.")
        self.elements /= norm
        return self
##########################################################################################


##########################################################################################
    def conjugate(self):
        return zulsquaternion(self.w, -self.x, -self.y, -self.z)
##########################################################################################


##########################################################################################
    def euler_to_quaternion(roll, pitch, yaw):
        # Conversion from euler angles (roll, pitch, yaw in RAD) to quaternion
        roll  = np.float32(roll)
        pitch = np.float32(pitch)
        yaw   = np.float32(yaw)
        
        cy = np.cos(yaw * 0.5)
        sy = np.sin(yaw * 0.5)
        cp = np.cos(pitch * 0.5)
        sp = np.sin(pitch * 0.5)
        cr = np.cos(roll * 0.5)
        sr = np.sin(roll * 0.5)

        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy

        return zulsquaternion(w, x, y, z)
##########################################################################################


##########################################################################################
    def quaternion_to_euler(self):
        # Conversion from quaternion to euler angles (roll, pitch, yaw in RAD)
        # Quaternion must be normalized
        sinr_cosp = 2 * (self.w * self.x + self.y * self.z)
        cosr_cosp = 1 - 2 * (self.x**2 + self.y**2)
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        sinp = 2 * (self.w * self.y - self.z * self.x)
        if np.abs(sinp) >= 1:
            pitch = np.sign(sinp) * (np.pi / 2)  # verwenden Sie π/2, um Gimbal Lock zu behandeln
        else:
            pitch = np.arcsin(sinp)

        siny_cosp = 2 * (self.w * self.z + self.x * self.y)
        cosy_cosp = 1 - 2 * (self.y**2 + self.z**2)
        yaw = np.arctan2(siny_cosp, cosy_cosp)

        return roll, pitch, yaw
##########################################################################################


##########################################################################################
    def derivative(self, q, omega):
        # Calculate the derivative of a quaternion
        # omega is the angular velocity vector, i.e. a tupel (omega_x, omega_y, omega_z)

        # Define omega as a quaternion
        q_omega = zulsquaternion(0, omega[0], omega[1], omega[2])

        # Derivative dq/dt = 1/2 * q * omega
        q_dot = 0.5 * q * q_omega
        
        return q_dot
##########################################################################################


##########################################################################################
    def integrate_euler(self, q, omega, dt):
        # Integration using Euler's method
        q_dot = self.derivative(q, omega)
        self += q_dot * dt
        
        return self.normalize()
##########################################################################################


##########################################################################################
    def integrate_rk4(self, q, omega, dt):
        # Integration using the Runge-Kutta 4th order method
        k1 = self.derivative(q, omega)
        k2 = self.derivative(q + dt/2 * k1, omega)
        k3 = self.derivative(q + dt/2 * k2, omega)
        k4 = self.derivative(q + dt   * k3, omega)
        
        self = q + dt/6 * (k1 + 2*k2 + 2*k3 + k4)

        return self.normalize()
##########################################################################################


##########################################################################################
    def __str__(self):
        return f"({self.w} + {self.x}i + {self.y}j + {self.z}k)"
##########################################################################################

