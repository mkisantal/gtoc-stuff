# Initial code from Marcus Maertens

# Some helpful functions for the GTOC preparation exercise
from pykep.core import epoch, DAY2SEC, MU_SUN, lambert_problem, propagate_lagrangian, fb_prop, AU, DEG2RAD
from pykep.planet import keplerian
from math import pi, cos, sin, acos, log
from scipy.linalg import norm

import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from pykep.orbit_plots import plot_planet, plot_lambert, plot_kepler

# orbital elements as copy and pasted: a,e,i,W,w,M
ENCELADUS_ELEM = [1.594457909000701E-03 * AU, 6.893905661052293E-03, 2.804154420040028E+01 * DEG2RAD, 1.695309235386006E+02 * DEG2RAD, 7.843853228362318E+01 * DEG2RAD, 2.900332219619573E+01 * DEG2RAD]
TETHYS_ELEM = [1.972039440701017E-03 * AU, 1.467193558689340E-03, 2.715868006547213E+01 * DEG2RAD, 1.681618900942828E+02 * DEG2RAD, 3.030260711406313E+02 * DEG2RAD, 3.214015637270131E+02 * DEG2RAD]
DIONE_ELEM = [2.523903548022649E-03 * AU, 2.898893113096072E-03, 2.807616131025873E+01 * DEG2RAD, 1.695239845489387E+02 * DEG2RAD, 6.450178107638715E+01 * DEG2RAD, 2.931671988838270E+01 * DEG2RAD]
RHEA_ELEM = [3.518754971601418E-03 * AU, 1.578062024637819E-03, 2.790835105559149E+01 * DEG2RAD, 1.627447833259773E+02 * DEG2RAD, 1.627447833259773E+02 * DEG2RAD, 1.848577690215480E+02 * DEG2RAD]
TITAN_ELEM = [8.166104015368233E-03 * AU, 2.862401678388353E-02, 2.769808447473740E+01 * DEG2RAD, 1.690711523205395E+02 * DEG2RAD, 1.746196276345883E+02 * DEG2RAD, 1.869169041332501E+02 * DEG2RAD]

# mass of the moons
MASS_ENCELADUS = 1.1e20
MASS_TETHYS = 6.2e20
MASS_DIONE = 1.1e21
MASS_RHEA = 2.3e21
MASS_TITAN = 1.35e23

# moon radii in m adapted from wiki (which showed diameter at this point)
RADIUS_ENCELADUS = 252000
RADIUS_TETHYS = 531000
RADIUS_DIONE = 561500
RADIUS_RHEA = 763500
RADIUS_TITAN = 2574500

# gravitional parameters
MU_ENCELADUS = 7.2094747e9
MU_TETHYS = 4.120904e10
MU_DIONE = 7.31113428e10
MU_RHEA = 1.53938857e11
MU_TITAN = 8.97797242e12

# central body is Saturn
MASS_SATURN = 5.6824e26
RADIUS_SATURN = 60268000
MU_SATURN = 3.7931187e16   # copied from wikipedia

# 2020-01-January, 00:00:00
REF_EPOCH = epoch(2458849.500000000, julian_date_type='jd')

# make the bodies
enceladus = keplerian(REF_EPOCH, ENCELADUS_ELEM, MU_SATURN, MU_ENCELADUS, RADIUS_ENCELADUS, RADIUS_ENCELADUS * 1.1, 'enceladus')
tethys = keplerian(REF_EPOCH, TETHYS_ELEM, MU_SATURN, MU_TETHYS, RADIUS_TETHYS, RADIUS_TETHYS * 1.1, 'tethys')
dione = keplerian(REF_EPOCH, DIONE_ELEM, MU_SATURN, MU_DIONE, RADIUS_DIONE, RADIUS_DIONE * 1.1, 'dione')
rhea = keplerian(REF_EPOCH, RHEA_ELEM, MU_SATURN, MU_RHEA, RADIUS_RHEA, RADIUS_RHEA * 1.1, 'rhea')
titan = keplerian(REF_EPOCH, TITAN_ELEM, MU_SATURN, MU_TITAN, RADIUS_TITAN, RADIUS_TITAN * 1.1, 'titan')

# different colors for plotting
pl2c = {'enceladus':'coral',
        'tethys':'seagreen',
        'dione': 'purple',
        'rhea': 'steelblue',
        'titan': 'firebrick',
        'pseudo': 'gray' }


class mga_1dsm_incipit:
    """
    Pimped up version of the original mga_1dsm. Uses tof encoding, allowing to constrain tof of each leg (super useful).

    The decision vector is: [t0, u, v, vinf, eta0, T0] + [beta, rp/rV, eta1, T1] + ...

    """
    def __init__(self, seq, t0, tof, vinf, add_vinf_dep=False, add_vinf_arr=False, multi_objective=False):
        """
        pykep.trajopt.mga_1dsm(seq = [jpl_lp('earth'), jpl_lp('venus'), jpl_lp('earth')], t0 = [epoch(0),epoch(1000)], tof = [1.0,5.0], vinf = [0.5, 2.5], multi_objective = False, add_vinf_dep = False, add_vinf_arr=True)

        - seq: list of pykep planets defining the encounter sequence (including the starting launch)
        - t0: list of two epochs defining the launch window
        - tof: list of [lower, upper] bounds for each leg
        - vinf: list of two floats defining the minimum and maximum allowed initial hyperbolic velocity (at launch), in m/sec
        - multi_objective: when True constructs a multiobjective problem (dv, T)
        - add_vinf_dep: when True the computed Dv includes the initial hyperbolic velocity (at launch)
        - add_vinf_arr: when True the computed Dv includes the final hyperbolic velocity (at the last planet)
        """

        # all planets need to have the same mu_central_body
        if ([r.mu_central_body for r in seq].count(seq[0].mu_central_body) != len(seq)):
            raise ValueError(
                'All planets in the sequence need to have exactly the same mu_central_body')
        
        # There should be upper and lower bounds for each leg of the sequence
        if len(tof) != (len(seq) - 1):
            raise ValueError(
                'Specify [lower, upper] for each leg of the sequence!') 
        for t in tof:
            if len(t) != 2:
                raise ValueError('Each leg needs to specify and upper and lower bound for tof!')         
        
        self.__add_vinf_dep = add_vinf_dep
        self.__add_vinf_arr = add_vinf_arr
        self.__n_legs = len(seq) - 1
        self._t0 = t0
        self._tof = tof
        self._vinf = vinf
        self._obj_dim = multi_objective + 1

        # We then define all planets in the sequence and the common central body gravity as data members
        self.seq = seq
        self.common_mu = seq[0].mu_central_body


    def get_nobj(self):
        return self._obj_dim


    def get_bounds(self):
        t0 = self._t0
        tof = self._tof
        vinf = self._vinf
        seq = self.seq
        
        # encoding: [t0, u, v, V_inf, eta0, T0] + [beta, rp1/rV1, eta1, T1]
        lb = [t0[0].mjd2000, 0.0, 0.0, vinf[0], 1e-5, 0]
        lb += [-2 * pi, 1.1, 1e-5, 0] * (self.__n_legs - 1)
        
        ub = [t0[1].mjd2000, 1.0, 1.0, vinf[1], 1.0 - 1e-5, 0]
        ub += [2 * pi, 30.0, 1.0 - 1e-5, 0] * (self.__n_legs - 1)
        
        # correct tof-bounds 
        for e, i in enumerate(range(5, len(lb)+1, 4)):
            lb[i], ub[i] = tof[e]
        
        # correct fly-by radius according to safe radius of planet
        for i, pl in enumerate(seq[1:-1]):
            lb[7 + 4 * i] = pl.safe_radius / pl.radius
        
        return (lb, ub)
    

    # computation of the objective function
    def _fitness_impl(self, x, logging=False, plotting=False, ax=None):
        # decode x
        t0, u, v, dep_vinf = x[:4]
        etas = x[4::4]
        T = x[5::4]
        betas = x[6::4]
        rps = x[7::4]
        
        # convert incoming velocity vector
        theta, phi = 2.0 * pi * u, acos(2.0 * v - 1.0) - pi / 2.0
        Vinfx = dep_vinf * cos(phi) * cos(theta)
        Vinfy = dep_vinf * cos(phi) * sin(theta)
        Vinfz = dep_vinf * sin(phi)

        # epochs and ephemerides of the planetary encounters
        t_P = list([None] * (self.__n_legs + 1))
        r_P = list([None] * (self.__n_legs + 1))
        v_P = list([None] * (self.__n_legs + 1))
        lamberts = list([None] * (self.__n_legs))
        v_outs = list([None] * (self.__n_legs))
        DV = list([0.0] * (self.__n_legs + 1))
        
        for i, planet in enumerate(self.seq):
            t_P[i] = epoch(t0 + sum(T[0:i]))
            r_P[i], v_P[i] = self.seq[i].eph(t_P[i])

        # first leg
        v_outs[0] = [a + b for a, b in zip(v_P[0], [Vinfx, Vinfy, Vinfz])]
        r, v = propagate_lagrangian(
            r_P[0], v_outs[0], etas[0] * T[0] * DAY2SEC, self.common_mu)

        # Lambert arc to reach seq[1]
        dt = (1.0 - etas[0]) * T[0] * DAY2SEC
        lamberts[0] = lambert_problem(r, r_P[1], dt, self.common_mu, False, 0)
        v_end_l = lamberts[0].get_v2()[0]
        v_beg_l = lamberts[0].get_v1()[0]

        # First DSM occuring at time eta0*T0
        DV[0] = norm([a - b for a, b in zip(v_beg_l, v)])

        # successive legs
        for i in range(1, self.__n_legs):
            # Fly-by
            v_outs[i] = fb_prop(v_end_l, v_P[i], rps[i-1] * self.seq[i].radius, betas[i-1], self.seq[i].mu_self)
            # s/c propagation before the DSM
            r, v = propagate_lagrangian(r_P[i], v_outs[i], etas[i] * T[i] * DAY2SEC, self.common_mu)
            # Lambert arc to reach next body
            dt = (1 - etas[i]) * T[i] * DAY2SEC
            lamberts[i] = lambert_problem(r, r_P[i + 1], dt, self.common_mu, False, 0)
            v_end_l = lamberts[i].get_v2()[0]
            v_beg_l = lamberts[i].get_v1()[0]
            # DSM occuring at time eta_i*T_i
            DV[i] = norm([a - b for a, b in zip(v_beg_l, v)])

        arr_vinf = norm([a - b for a, b in zip(v_end_l, v_P[-1])])

        # last Delta-v
        if self.__add_vinf_arr:
            DV[-1] = arr_vinf

        if self.__add_vinf_dep:
            DV[0] += dep_vinf

        # pretty printing
        if logging:
            print("First leg: {} to {}".format(self.seq[0].name, self.seq[1].name))
            print("Departure: {0} ({1:0.6f} mjd2000)".format(t_P[0], t_P[0].mjd2000))
            print("Duration: {0:0.6f}d".format(T[0]))
            print("VINF: {0:0.3f}m/s".format(dep_vinf))
            print("DSM after {0:0.6f}d".format(etas[0] * T[0]))
            print("DSM magnitude: {0:0.6f}m/s".format(DV[0]))

            for i in range(1, self.__n_legs):
                print("\nleg {}: {} to {}".format(i + 1, self.seq[i].name, self.seq[i + 1].name))
                print("Duration: {0:0.6f}d".format(T[i]))
                print("Fly-by epoch: {0} ({1:0.6f} mjd2000)".format(t_P[i], t_P[i].mjd2000))
                print("Fly-by radius: {0:0.6f} planetary radii".format(rps[i-1]))
                print("DSM after {0:0.6f}d".format(etas[i] * T[i]))
                print("DSM magnitude: {0:0.6f}m/s".format(DV[i]))

            print("\nArrival at {}".format(self.seq[-1].name))
            print("Arrival epoch: {0} ({1:0.6f} mjd2000)".format(t_P[-1], t_P[-1].mjd2000))
            print("Arrival Vinf: {0:0.3f}m/s".format(arr_vinf))
            print("Total mission time: {0:0.6f}d ({1:0.3f} years)".format(sum(T), sum(T) / 365.25))

        # plotting
        if plotting:
            ax.scatter(0, 0, 0, color='chocolate')
            for i, planet in enumerate(self.seq):
                plot_planet(planet, t0=t_P[i], color=pl2c[planet.name], legend=True, units=AU, ax=ax)
            for i in range(0, self.__n_legs):                
                plot_kepler(r_P[i], v_outs[i], etas[i] * T[i] * DAY2SEC, self.common_mu, N=100, color='b', legend=False, units=AU, ax=ax)            
            for l in lamberts:
                plot_lambert(l, sol=0, color='r', legend=False, units=AU, N=1000, ax=ax)

        # returning building blocks for objectives
        return (DV, T, arr_vinf)


    def fitness(self, x):
        """ this gives you the fitness, without any excess baggage """
        DV, T, arr_vinf = self._fitness_impl(x)
        if self._obj_dim == 1:
            return (sum(DV),)
        else:
            return (sum(DV), sum(T))
            
    
    def pretty(self, x):
        """ pretty printing trajectory information  """
        _ = self._fitness_impl(x, True)


    def plot(self, x, ax=None):
        """ plots the trajectory """
        if ax is None:
            mpl.rcParams['legend.fontsize'] = 10
            fig = plt.figure()
            axis = fig.gca(projection='3d')
        else:
            axis = ax        
        
        _ = self._fitness_impl(x, logging=False, plotting=True, ax=axis)
        return axis
        

    def get_extra_info(self):
        """ for the particularly curious... """
        return ("\n\t Sequence: " + [pl.name for pl in self.seq].__repr__() +
                "\n\t Add launcher vinf to the objective?: " + self.__add_vinf_dep.__repr__() +
                "\n\t Add final vinf to the objective?: " + self.__add_vinf_arr.__repr__())
