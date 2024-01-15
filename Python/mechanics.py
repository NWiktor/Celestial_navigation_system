def rk4(f, t0, y0, h):
    """ A Runge-Kutta módszer egy függvény értékét közelíti egy lépéssel később.

    Adott egy f(t, y(t)) függvény, ahol ismert egy kezdeti érték: y(t0)=y
    f - keresett függvény
    t0 - kezdeti idő
    y0 - kezdeti fv. érték t0 időpillanatban
    h - lépésköz
    https://lpsa.swarthmore.edu/NumInt/NumIntFourth.html
    https://www.youtube.com/watch?v=TzX6bg3Kc0E&list=PLOIRBaljOV8hBJS4m6brpmUrncqkyXBjB&index=5
    """
    k1 = f(t0, y0)
    k2 = f(t0 + 0.5 * h, y0 + 0.5 * k1 * h)
    k3 = f(t0 + 0.5 * h, y0 + 0.5 * k2 * h)
    k4 = f(t0 + h, y0 + k3 * h)

    # Returns integral of function
    return y0 + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)

# TODO: gyorsulásvektor (f(y0) kiszámítása, mivel ismerem a pozíciót és a sebességet (y0)
# TODO: a gyorsulásvektor integrálása RK4-el, így megkapom a sebességvektort
# TODO: a sebessévektor integrálása RK4-el, így megkapom a pozícióvektort
# TODO: mivel a f(y0) egyben a sebességvektort is visszaadja, ezért az RK-4 is kétszer tud integrálni - egy lépésben


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
