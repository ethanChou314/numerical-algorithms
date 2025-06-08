import numpy as np


def spline_cubic(x, xdata, ydata):
	"""
	Parameters:
		x: 
	"""
	m = compute_derivatives()
	y = []
	for i in range(1, xdata.size):
		x1 = xdata[i-1]
		x2 = xdata[i]
		y1 = ydata[i-1]
		y2 = ydata[i]

		# get interpolation values in this range:
		mask = (x1 <= x) & (x < x2)
		x_in_range = x[mask]

		# interpolate this range:
		y_in_range = p(x_in_range, x1, x2, 
						y1, y2, m1, m2)
		y = np.append(y, y_in_range)

	return y


def p(x, x1, x2, y1, y2, m1, m2):
	"""
	The interpolation function over [x1, x2]

	Parameters:
		x [array[float]]: x-coord of interpolated values
		x1 [float]:  
	"""
	coeffs = solve_coeffs(x1, x2, y1, y2, m1, m2)
	power = np.arange(coeffs.size)
	return np.sum(coeffs * (x-x1)**power)


def solve_coeffs(x1, x2, y1, y2, m1, m2):
	"""
	Solves for coefficients of cubic equation over [x1, x2]
	based on the constraints (where p is the polynomial 
	for interpolation):
		1. p(x1) = f(x1)
		2. p(x2) = f(x2)
		3. p'(x1) = f'(x1)
		4. p'(x2) = f'(x2)

	Parameters:
		x1 [float]: start of interval
		x2 [float]: end of interval
		y1 [float]: f(x1)
		y2 [float]: f(x2)
		m1 [float]: f'(x1)
		m2 [float]: f'(x2)

	Returns:
		coeffs [array[float]]: [a, b, c, d]
		where p(x) = a + bx + cx^2 + dx^3
	"""
	# if Ax = b and x is our solution:
	dx = x2 - x1
	A = np.array([[1, 0, 0, 0],
				  [1, dx, dx**2, dx**3],
				  [0, 1, 0, 0],
				  [0, 1, 2*dx, 3*dx**2]])
	b = np.array([y1, y2, m1, m2])

	# solve:
	coeffs = np.linalg.solve(A, b)
	return coeffs


def compute_derivatives()
