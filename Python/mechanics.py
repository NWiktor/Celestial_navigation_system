def rk4(func, t0, y0, h, *args):
    """ The fourth-order Runge-Kutta method approximates the solution (function) of a first-order ODE.

    Adott egy y'(t) = f(t, y(t), *args) ODE függvény, és ismert a megoldás fv. egy kezdeti értéke: y(t0)=y0
    func - ODE függvény
    t0 - kezdeti idő
    y0 - kezdeti fv. érték t0 időpillanatban
    h - lépésköz
    https://lpsa.swarthmore.edu/NumInt/NumIntFourth.html
    https://www.youtube.com/watch?v=TzX6bg3Kc0E&list=PLOIRBaljOV8hBJS4m6brpmUrncqkyXBjB&index=5
    """
    k1 = func(t0, y0, *args)
    k2 = func(t0 + 0.5 * h, y0 + 0.5 * k1 * h, *args)
    k3 = func(t0 + 0.5 * h, y0 + 0.5 * k2 * h, *args)
    k4 = func(t0 + h, y0 + k3 * h, *args)

    # Returns y1 value, which is the approximation of the y(t) function at t1:
    # This is basically the velocity-vector
    return y0 + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)

# TODO: gyorsulásvektor (f(y1) kiszámítása, mivel ismerem a pozíciót és a sebességet (y0)
# TODO: a gyorsulásvektor közelítése RK4-el, így megkapom a sebességvektort
# TODO: a sebessévektor integrálása RK4-el, így megkapom a pozícióvektort
# TODO: mivel a f(y0) egyben a sebességvektort is visszaadja, ezért az RK-4 is kétszer tud integrálni - egy lépésben


# Include guard
if __name__ == '__main__':
    pass

# def rk4_ks( f, t, y, h ):
# 	'''
# 	Calculate all RK4 k values
# 	'''
# 	k1 = f( t, y )
# 	k2 = f( t + 0.5 * h, y + 0.5 * k1 * h )
# 	k3 = f( t + 0.5 * h, y + 0.5 * k2 * h )
# 	k4 = f( t +       h, y +       k3 * h )
# 	kt = 1 / 6.0 * ( k1 + 2 * k2 + 2 * k3 + k4 )
# 	return k1, k2, k3, k4, kt
#
#
# def rk4_step( f, t, y, h ):
# 	'''
# 	Take one RK4 step
# 	'''
# 	k1 = f( t, y )
# 	k2 = f( t + 0.5 * h, y + 0.5 * k1 * h )
# 	k3 = f( t + 0.5 * h, y + 0.5 * k2 * h )
# 	k4 = f( t +       h, y +       k3 * h )
#
# 	return y + h / 6.0 * ( k1 + 2 * k2 + 2 * k3 + k4 )
#
#
# def two_body_ode( t, state, mu = pd.earth[ 'mu' ] ):
# 	# state = [ rx, ry, rz, vx, vy, vz ]
#
# 	r = state[ :3 ]
# 	a = -mu * r / np.linalg.norm( r ) ** 3
#
# 	return np.array( [
# 		state[ 3 ], state[ 4 ], state[ 5 ],
# 		    a[ 0 ],     a[ 1 ],     a[ 2 ] ] )
#
#
# def diffy_q( self, et, state ):
# 		rx, ry, rz, vx, vy, vz, mass = state
# 		r         = np.array( [ rx, ry, rz ] )
# 		mass_dot  = 0.0
# 		state_dot = np.zeros( 7 )
# 		et       += self.et0
#
# 		a = -r * self.cb[ 'mu' ] / nt.norm( r ) ** 3
#
# 		for pert in self.orbit_perts_funcs:
# 			a += pert( et, state )
#
# 		state_dot[ :3  ] = [ vx, vy, vz ]
# 		state_dot[ 3:6 ] = a
# 		state_dot[ 6   ] = mass_dot
# 		return state_dot
