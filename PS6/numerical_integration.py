import numpy as np
from tabulate import tabulate

def func(y):
	"""Defining the function F for our differential equation"""
	return float(3*y - 2)


# We start with Euler's Method:
def euler(func, x0, y0, N, x):
	"""Using the Euler method to numerically approximate a solution to our equation with F func,
	intial conditions x0 and y0, N steps and final x value x"""
	h = float((x-x0)/float(N))
	x_arr = np.linspace(x0, x, num=N+1)
	y_arr = np.linspace(x0, x, num=N+1)
	y_arr[0] = y0
	y_ac = np.linspace(x0, x, num=N+1)
	y_ac[0] = y0
	err = np.linspace(x0, x, num=N+1)
	err[0] = 0
	xy_arr = np.zeros((N+1, 4))
	xy_arr[0, 0], xy_arr[0, 1], xy_arr[0,2], xy_arr[0, 3] = x0, y0, y0, 0

	for j in range(N+1)[1:]:
		y_arr[j] = y_arr[j-1] + float(h)*float(func(y_arr[j-1]))
		y_ac[j] = float((np.e**(3*x_arr[j]) + 2)/3)
		err[j] = float(y_ac[j] - y_arr[j])
		xy_arr[j, 0], xy_arr[j, 1], xy_arr[j, 2], xy_arr[j, 3] 	= x_arr[j], y_arr[j], y_ac[j], err[j]

	return xy_arr


# We now use the Euler Trapezoid method
def euler_trap(func, x0, y0, N, x):
	"""Using the Euler trapezoid method to numerically approximate a solution to our equation with F func,
	intial conditions x0 and y0, N steps and final x value x"""
	h = float((x-x0)/float(N))
	x_arr = np.linspace(x0, x, num=N+1)
	y_arr = np.linspace(x0, x, num=N+1)
	y_arr[0] = y0
	y_ac = np.linspace(x0, x, num=N+1)
	y_ac[0] = y0
	err = np.linspace(x0, x, num=N+1)
	err[0] = 0
	xy_arr = np.zeros((N+1, 4))
	xy_arr[0, 0], xy_arr[0, 1], xy_arr[0,2], xy_arr[0, 3] = x0, y0, y0, 0

	for j in range(N+1)[1:]:
		yj1 = y_arr[j-1] + float(h)*float(func(y_arr[j-1]))
		y_arr[j] = y_arr[j-1] + float(h/2)*(float(func(y_arr[j-1])) + float(func(yj1)))
		y_ac[j] = float((np.e**(3*x_arr[j]) + 2)/3)
		err[j] = float(y_ac[j] - y_arr[j])
		xy_arr[j, 0], xy_arr[j, 1], xy_arr[j, 2], xy_arr[j, 3] 	= x_arr[j], y_arr[j], y_ac[j], err[j]
	
	return xy_arr


# We now use the Runge-Kutta method
def runge_kutta(func, x0, y0, N, x):
	"""Using the Runge-Kutta method to numerically approximate a solution to our equation with F func,
	intial conditions x0 and y0, N steps and final x value x"""
	h = float((x-x0)/float(N))
	x_arr = np.linspace(x0, x, num=N+1)
	y_arr = np.linspace(x0, x, num=N+1)
	y_arr[0] = y0
	y_ac = np.linspace(x0, x, num=N+1)
	y_ac[0] = y0
	err = np.linspace(x0, x, num=N+1)
	err[0] = 0
	xy_arr = np.zeros((N+1, 4))
	xy_arr[0, 0], xy_arr[0, 1], xy_arr[0,2], xy_arr[0, 3] = x0, y0, y0, 0

	for j in range(N+1)[1:]:
                kj1 = float(func(y_arr[j-1]))
                kj2 = float(func(y_arr[j-1] + float(h/2)*kj1))
                kj3 = float(func(y_arr[j-1] + float(h/2)*kj2))
                kj4 = float(func(y_arr[j-1] + float(h)*kj3))
                y_arr[j] = y_arr[j-1] + float(h)*((kj1 + 2*kj2 + 2*kj3 + kj4)/6)
		y_ac[j] = float((np.e**(3*x_arr[j]) + 2)/3)
		err[j] = float(y_ac[j] - y_arr[j])
		xy_arr[j, 0], xy_arr[j, 1], xy_arr[j, 2], xy_arr[j, 3] 	= x_arr[j], y_arr[j], y_ac[j], err[j]

        return xy_arr


xval = euler(func, 0, 1, 200, 10)[:,0]
yval = euler(func, 0, 1, 200, 10)[:,2]
euler_val = euler(func, 0, 1, 200, 10)[:,1]
euler_err = euler(func, 0, 1, 200, 10)[:,3]
trap_val = euler_trap(func, 0, 1, 200, 10)[:,1]
trap_err = euler_trap(func, 0, 1, 200, 10)[:,3]
runge_val = runge_kutta(func, 0, 1, 200, 10)[:,1]
runge_err = runge_kutta(func, 0, 1, 200, 10)[:,3]

comparison = np.c_[ xval, yval, euler_val, euler_err, trap_val, trap_err, runge_val, runge_err ]

print(tabulate(comparison, floatfmt='.6e'))

