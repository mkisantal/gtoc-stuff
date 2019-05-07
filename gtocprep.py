# Some helpful functions for the GTOC preparation exercise
from pykep.core import epoch, DAY2SEC, MU_SUN, lambert_problem, propagate_lagrangian, fb_prop, AU, DEG2RAD, G0, ic2par
from pykep.planet import keplerian
from math import pi, cos, sin, acos, log, exp
from scipy.linalg import norm

import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from pykep.orbit_plots import plot_planet, plot_lambert, plot_kepler

# orbital elements as copy and pasted: a,e,i,W,w,M
ENCELADUS_ELEM = [1.594457909000701E-03 * AU, 6.893905661052293E-03, 2.804154420040028E+01 * DEG2RAD,
                  1.695309235386006E+02 * DEG2RAD, 7.843853228362318E+01 * DEG2RAD, 2.900332219619573E+01 * DEG2RAD]
TETHYS_ELEM = [1.972039440701017E-03 * AU, 1.467193558689340E-03, 2.715868006547213E+01 * DEG2RAD,
               1.681618900942828E+02 * DEG2RAD, 3.030260711406313E+02 * DEG2RAD, 3.214015637270131E+02 * DEG2RAD]
DIONE_ELEM = [2.523903548022649E-03 * AU, 2.898893113096072E-03, 2.807616131025873E+01 * DEG2RAD,
              1.695239845489387E+02 * DEG2RAD, 6.450178107638715E+01 * DEG2RAD, 2.931671988838270E+01 * DEG2RAD]
RHEA_ELEM = [3.518754971601418E-03 * AU, 1.578062024637819E-03, 2.790835105559149E+01 * DEG2RAD,
             1.627447833259773E+02 * DEG2RAD, 1.627447833259773E+02 * DEG2RAD, 1.848577690215480E+02 * DEG2RAD]
TITAN_ELEM = [8.166104015368233E-03 * AU, 2.862401678388353E-02, 2.769808447473740E+01 * DEG2RAD,
              1.690711523205395E+02 * DEG2RAD, 1.746196276345883E+02 * DEG2RAD, 1.869169041332501E+02 * DEG2RAD]

# mass of the moons
MASS_ENCELADUS = 1.1e20
MASS_TETHYS = 6.2e20
MASS_DIONE = 1.1e21
MASS_RHEA = 2.3e21
MASS_TITAN = 1.35e23

# moon radii in m adapted from wiki
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
MU_SATURN = 3.7931187e16  # copied from wikipedia

# specific impulse
ISP = 308

# constants related to mass
M_WET = 5000  # kg
M_DRY = 3000  # kg
DV_BUDGET = ISP * G0 * log(M_WET / M_DRY)

# additive constant for violation of constraints
DV_PENALTY = 200000

# constants related to time
T_START, T_END = 20000, 24000  # MJD2000

# 2020-01-January, 00:00:00
REF_EPOCH = epoch(2458849.500000000, julian_date_type='jd')

# make the bodies
enceladus = keplerian(REF_EPOCH, ENCELADUS_ELEM, MU_SATURN, MU_ENCELADUS, RADIUS_ENCELADUS, RADIUS_ENCELADUS * 1.1,
                      'enceladus')
tethys = keplerian(REF_EPOCH, TETHYS_ELEM, MU_SATURN, MU_TETHYS, RADIUS_TETHYS, RADIUS_TETHYS * 1.1, 'tethys')
dione = keplerian(REF_EPOCH, DIONE_ELEM, MU_SATURN, MU_DIONE, RADIUS_DIONE, RADIUS_DIONE * 1.1, 'dione')
rhea = keplerian(REF_EPOCH, RHEA_ELEM, MU_SATURN, MU_RHEA, RADIUS_RHEA, RADIUS_RHEA * 1.1, 'rhea')
titan = keplerian(REF_EPOCH, TITAN_ELEM, MU_SATURN, MU_TITAN, RADIUS_TITAN, RADIUS_TITAN * 1.1, 'titan')

# different colors for plotting
pl2c = {'enceladus': 'coral',
        'tethys': 'seagreen',
        'dione': 'purple',
        'rhea': 'steelblue',
        'titan': 'firebrick',
        'pseudo': 'gray'}


def mass_spent(dv, m_total, isp=ISP):
    veff = isp * G0
    mf = m_total / exp(dv / veff)
    return m_total - mf


class MgaDsm1base:
    """
    Uses tof encoding. Can better be used as a base class later than the original version.

    The decision vector is: [t0, u, v, vinf, eta0, T0] + [beta, rp/rV, eta1, T1] + ...

    """

    def __init__(self, seq, t0, tof, etas, vinf, add_vinf_dep=False, add_vinf_arr=False, multi_objective=False,
                 cw=False):
        """

        - seq: list of pykep planets defining the encounter sequence. Needs a placeholder as a starting position (will get overwritten.)
        - t0: list of two epochs defining the launch window
        - tof: list of [lower, upper] bounds for each leg
        - etas: list of [lower, upper] bounds for each leg
        - vinf: list of two floats defining the min and max allowed initial hyperbolic velocity (at launch), in m/s
        - multi_objective: when True constructs a multiobjective problem (dv, T)
        - add_vinf_dep: when True the computed Dv includes the initial hyperbolic velocity (at launch)
        - add_vinf_arr: when True the computed Dv includes the final hyperbolic velocity (at the last planet)
        """

        # all planets need to have the same mu_central_body
        if ([r.mu_central_body for r in seq].count(seq[0].mu_central_body) != len(seq)):
            raise ValueError(
                'All planets in the sequence need to have exactly the same mu_central_body')

        # There should be upper and lower bounds for each leg of the sequence
        # if len(tof) != (len(seq) - 1):
        #     raise ValueError(
        #         'Specify [lower, upper] for each leg of the sequence!')
        # for t in tof:
        #     if len(t) != 2:
        #         raise ValueError('Each leg needs to specify and upper and lower bound for tof!')

        self._add_vinf_dep = add_vinf_dep
        self._add_vinf_arr = add_vinf_arr
        self._n_legs = len(seq) - 1
        self._t0 = t0
        self._tof = tof
        self._etas = etas
        self._vinf = vinf
        self._obj_dim = multi_objective + 1
        self.cw = cw

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
        lb += [-2 * pi, 1.1, 1e-5, 0] * (self._n_legs - 1)

        ub = [t0[1].mjd2000, 1.0, 1.0, vinf[1], 1.0 - 1e-5, 0]
        ub += [2 * pi, 30.0, 1.0 - 1e-5, 0] * (self._n_legs - 1)

        # correct tof-bounds
        for e, i in enumerate(range(5, len(lb) + 1, 4)):
            lb[i], ub[i] = tof[e]

        # correct etas
        if self._etas is not None:
            for e, i in enumerate(range(4, len(lb) + 1, 4)):
                lb[i], ub[i] = self._etas[e]

        # correct fly-by radius according to safe radius of planet
        for i, pl in enumerate(seq[1:-1]):
            lb[7 + 4 * i] = pl.safe_radius / pl.radius

        return (lb, ub)

    def decode_state_eval_fitness(self, x, logging=False, plotting=False, ax=None):

        """ Intermediate step, helps including initial orbit parameters in child class. """

        t0, u, v, dep_vinf = x[:4]
        etas = x[4::4]
        T = x[5::4]
        betas = x[6::4]
        rps = x[7::4]

        decoded = [t0, u, v, dep_vinf, etas, T, betas, rps]

        return self._fitness_impl(decoded, logging=logging, plotting=plotting, ax=ax)

    @staticmethod
    def periapsis_passed(E0, E, delta_t, period):

        """ Given two anomaly, the time and the period, is the spacecraft passing the periapsis? """

        if delta_t > period:
            return True  # over a whole period we pass it for sure

        if E0 > 0:  # spacecraft flying away from planet
            return 0 < E < E0
        else:  # flying towards planet
            return E > 0 or (delta_t > period / 2)

    def check_distance(self, r0, v0, t_start, t_end, safe_radius=RADIUS_SATURN * 2):

        """ Is the periapsis closer than  safe radius, and if so are we passing periapsis? """

        # get orbital parameters
        a, e, _, _, _, E0 = ic2par(r0, v0, self.common_mu)
        rp = a * (1 - e)

        if rp > safe_radius:  # we are safe if periapsis is large enough
            return False
        else:
            # now we need to check if we actually pass the periapsis in this conic

            # propagate flight
            delta_t = t_end - t_start
            r, v = propagate_lagrangian(r0, v0, delta_t * DAY2SEC, self.common_mu)

            # get current orbital period [days]
            period = 2 * pi * (a ** 3 / self.common_mu) ** .5 / DAY2SEC

            # calculate new anomaly
            _, _, _, _, _, E = ic2par(r, v, self.common_mu)

            # check if we actually pass close to planet
            return self.periapsis_passed(E0, E, delta_t, period)

    def _fitness_impl(self, decoded_x, logging=False, plotting=False, ax=None):

        """ Computation of the objective function. """

        saturn_distance_violated = 0

        # decode x
        t0, u, v, dep_vinf, etas, T, betas, rps = decoded_x

        # convert incoming velocity vector
        theta, phi = 2.0 * pi * u, acos(2.0 * v - 1.0) - pi / 2.0
        Vinfx = dep_vinf * cos(phi) * cos(theta)
        Vinfy = dep_vinf * cos(phi) * sin(theta)
        Vinfz = dep_vinf * sin(phi)

        # epochs and ephemerides of the planetary encounters
        t_P = list([None] * (self._n_legs + 1))
        r_P = list([None] * (self._n_legs + 1))
        v_P = list([None] * (self._n_legs + 1))
        lamberts = list([None] * (self._n_legs))
        v_outs = list([None] * (self._n_legs))
        DV = list([0.0] * (self._n_legs + 1))

        for i, planet in enumerate(self.seq):
            t_P[i] = epoch(t0 + sum(T[0:i]))
            r_P[i], v_P[i] = self.seq[i].eph(t_P[i])

        # first leg
        v_outs[0] = [Vinfx, Vinfy, Vinfz]  # bug fixed

        # check first leg up to DSM
        saturn_distance_violated += self.check_distance(r_P[0], v_outs[0], t0, etas[0] * T[0])
        r, v = propagate_lagrangian(
            r_P[0], v_outs[0], etas[0] * T[0] * DAY2SEC, self.common_mu)

        # Lambert arc to reach seq[1]
        dt = (1.0 - etas[0]) * T[0] * DAY2SEC
        lamberts[0] = lambert_problem(r, r_P[1], dt, self.common_mu, self.cw, 0)
        v_end_l = lamberts[0].get_v2()[0]
        v_beg_l = lamberts[0].get_v1()[0]

        # First DSM occuring at time eta0*T0
        DV[0] = norm([a - b for a, b in zip(v_beg_l, v)])
        # checking first leg after DSM
        saturn_distance_violated += self.check_distance(r, v_beg_l, etas[0] * T[0], T[0])

        # successive legs
        for i in range(1, self._n_legs):
            # Fly-by
            v_outs[i] = fb_prop(v_end_l, v_P[i], rps[i - 1] * self.seq[i].radius, betas[i - 1], self.seq[i].mu_self)
            # checking next leg up to DSM
            saturn_distance_violated += self.check_distance(r_P[i], v_outs[i], T[i - 1], etas[i] * T[i])
            # s/c propagation before the DSM
            r, v = propagate_lagrangian(r_P[i], v_outs[i], etas[i] * T[i] * DAY2SEC, self.common_mu)
            # Lambert arc to reach next body
            dt = (1 - etas[i]) * T[i] * DAY2SEC
            lamberts[i] = lambert_problem(r, r_P[i + 1], dt, self.common_mu, self.cw, 0)
            v_end_l = lamberts[i].get_v2()[0]
            v_beg_l = lamberts[i].get_v1()[0]
            # DSM occuring at time eta_i*T_i
            DV[i] = norm([a - b for a, b in zip(v_beg_l, v)])
            # checking next leg after DSM
            saturn_distance_violated += self.check_distance(r, v_beg_l, etas[i] * T[i], T[i])

        # single dv penalty for now
        if saturn_distance_violated > 0:
            DV[-1] += DV_PENALTY

        arr_vinf = norm([a - b for a, b in zip(v_end_l, v_P[-1])])

        # last Delta-v
        if self._add_vinf_arr:
            DV[-1] = arr_vinf

        if self._add_vinf_dep:
            DV[0] += dep_vinf

        # pretty printing
        if logging:
            print("First leg: {} to {}".format(self.seq[0].name, self.seq[1].name))
            print("Departure: {0} ({1:0.6f} mjd2000)".format(t_P[0], t_P[0].mjd2000))
            print("Duration: {0:0.6f}d".format(T[0]))
            print("VINF: {0:0.3f}m/s".format(dep_vinf))
            print("DSM after {0:0.6f}d".format(etas[0] * T[0]))
            print("DSM magnitude: {0:0.6f}m/s".format(DV[0]))

            for i in range(1, self._n_legs):
                print("\nleg {}: {} to {}".format(i + 1, self.seq[i].name, self.seq[i + 1].name))
                print("Duration: {0:0.6f}d".format(T[i]))
                print("Fly-by epoch: {0} ({1:0.6f} mjd2000)".format(t_P[i], t_P[i].mjd2000))
                print("Fly-by radius: {0:0.6f} planetary radii".format(rps[i - 1]))
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
            for i in range(0, self._n_legs):
                plot_kepler(r_P[i], v_outs[i], etas[i] * T[i] * DAY2SEC, self.common_mu, N=100, color='b', legend=False,
                            units=AU, ax=ax)
            for l in lamberts:
                plot_lambert(l, sol=0, color='r', legend=False, units=AU, N=1000, ax=ax)

        # returning building blocks for objectives
        return (DV, T, arr_vinf, lamberts)

    def fitness(self, x):
        """ this gives you the fitness, without any excess baggage """
        # DV, T, arr_vinf, lamberts = self._fitness_impl(x)
        DV, T, arr_vinf, lamberts = self.decode_state_eval_fitness(x)
        if self._obj_dim == 1:
            return (sum(DV),)
        else:
            return (sum(DV), sum(T))

    def get_patching_conditions(self, x):
        """ returns the final epoch of the trajectory, the last ending point and the outgoing velocity """
        # DV, T, arr_vinf, lamberts = self._fitness_impl(x)
        DV, T, arr_vinf, lamberts = self.decode_state_eval_fitness(x)
        v = lamberts[-1].get_v2()[0]
        return (epoch(x[0] + sum(T)), self.seq[-1], v)

    def pretty(self, x):
        """ pretty printing trajectory information  """
        _ = self.decode_state_eval_fitness(x, True)

    def plot(self, x, ax=None):
        """ plots the trajectory """
        if ax is None:
            mpl.rcParams['legend.fontsize'] = 10
            fig = plt.figure()
            axis = fig.gca(projection='3d')
        else:
            axis = ax

        _ = self.decode_state_eval_fitness(x, logging=False, plotting=True, ax=axis)
        return axis

    def get_extra_info(self):
        """ for the particularly curious... """
        return ("\n\t Sequence: " + [pl.name for pl in self.seq].__repr__() +
                "\n\t Add launcher vinf to the objective?: " + self._add_vinf_dep.__repr__() +
                "\n\t Add final vinf to the objective?: " + self._add_vinf_arr.__repr__())


class MgaDsm1Start(MgaDsm1base):
    """  Adding initial anomaly as an extra parameter to the chromosome.  """

    def __init__(self, seq, t0, tof, etas, vinf, anomaly,
                 add_vinf_dep=False, add_vinf_arr=False, multi_objective=False, cw=False):
        super(MgaDsm1Start, self).__init__(seq, t0, tof, etas, vinf,
                                           add_vinf_dep, add_vinf_arr, multi_objective, cw)
        self._anomaly = anomaly

    def start_planet(self, anomaly):
        a = 1000 * RADIUS_SATURN
        e = 0.0
        W, M = 1.690711523205395E+02 * DEG2RAD, 1.869169041332501E+02 * DEG2RAD
        i = 2.769808447473740E+01 * DEG2RAD  # inclination of Titan.
        w = anomaly * 360.0 * DEG2RAD
        pseudo = keplerian(REF_EPOCH, [a, e, i, W, w, M], MU_SATURN, 1, 1, 1 * 1.1, 'pseudo')
        return pseudo

    def get_bounds(self):
        (lb1, ub1) = super(MgaDsm1Start, self).get_bounds()

        lb = [self._anomaly[0]] + lb1
        ub = [self._anomaly[1]] + ub1

        return lb, ub

    def decode_state_eval_fitness(self, x, logging=False, plotting=False, ax=None):
        """ Including initial orbit params in the encoding, overwriting parent method. """

        anomaly, t0, u, v, dep_vinf = x[:5]
        etas = x[5::4]
        T = x[6::4]
        betas = x[7::4]
        rps = x[8::4]

        pseudo = self.start_planet(anomaly)
        self.seq[0] = pseudo  # overwriting the initial position!

        decoded = [t0, u, v, dep_vinf, etas, T, betas, rps]

        return self._fitness_impl(decoded, logging=logging, plotting=plotting, ax=ax)


class MgaDsm1Full(MgaDsm1Start):
    """ One UDP from start to impact. """

    def __init__(self, seq, t0, tof, vinf, anomaly,
                 etas=None, add_vinf_dep=False, add_vinf_arr=False, multi_objective=False,
                 impact=False, mass=None, isp=None, cw=False):
        super(MgaDsm1Full, self).__init__(seq, t0, tof, etas, vinf, anomaly,
                                          add_vinf_dep, add_vinf_arr, multi_objective, cw)

        self._mass = mass
        self._isp = isp
        self._dv_budget = isp * G0 * log(mass[1] / mass[0])
        self.impact = impact

    def fitness(self, x, impact_energy_multiplier=1e-9):

        """ Fitness with impact: if deltaV is fine, impact energy is optimized. """

        DV, T, arr_vinf, lamberts = self.decode_state_eval_fitness(x)
        # compute the final mass at impact
        sum_dv = sum(DV)

        if sum_dv > self._dv_budget or not self.impact:
            # unless we lower our DV, we cannot impact: hence, we optimize
            # for dv in this case:
            return [sum_dv]
        else:
            m_dry, m_tot = self._mass
            veff = self._isp * G0
            m_prop = m_tot / exp(sum_dv / veff)
            m_f = m_tot - m_prop

        # Note that we want to maximize impact energy, so we minimize -J
        J = -arr_vinf ** 2 * m_f * impact_energy_multiplier

        objective = J
        if self._obj_dim == 1:
            return [objective]
        else:
            return [objective, sum(T)]


class ImpactLeg(MgaDsm1Full):

    """ Optimizing the last leg: the impact to the final planet of the sequence. """

    def __init__(self, chromosome, T_impact, eta_impact, seq, mass, isp):
        self.fixed = self.decode_init_chromosome(chromosome)
        super(ImpactLeg, self).__init__(seq, t0=None, tof=None, vinf=None, anomaly=None,
                                        impact=True, mass=mass, isp=isp)
        self.T_impact = T_impact
        self.eta_impact = eta_impact

    def get_bounds(self):
        lb = [self.eta_impact[0], self.T_impact[0]]
        ub = [self.eta_impact[1], self.T_impact[1]]
        return lb, ub

    @staticmethod
    def decode_init_chromosome(x):
        # encoding: [t0, u, v, V_inf, eta0, T0] + [beta, rp1/rV1, eta1, T1]
        anomaly, t0, u, v, dep_vinf = x[:5]
        etas = x[5::4]
        T = x[6::4]
        betas = x[7::4]
        rps = x[8::4]

        # getting the fixed part of the chromosome (tour up to last leg)
        fixed = {'anomaly': anomaly,
                 't0': t0, 'u': u, 'v': v, 'dep_vinf': dep_vinf,
                 'etas': list(etas[:-1]),
                 'T': list(T[:-1]),
                 'betas': betas,
                 'rps': rps}

        return fixed

    def decode_state_eval_fitness(self, x, logging=False, plotting=False, ax=None):
        """ Getting the fixed part, adding the encoded parameters. """

        anomaly, t0, u, v = self.fixed['anomaly'], self.fixed['t0'], self.fixed['u'], self.fixed['v']
        dep_vinf = self.fixed['dep_vinf']
        etas = list(self.fixed['etas']) + [x[0]]
        T = list(self.fixed['T']) + [x[1]]
        betas = self.fixed['betas']
        rps = self.fixed['rps']

        pseudo = self.start_planet(anomaly)
        self.seq[0] = pseudo  # overwriting the initial position!

        decoded = [t0, u, v, dep_vinf, etas, T, betas, rps]

        return self._fitness_impl(decoded, logging=logging, plotting=plotting, ax=ax)

    def get_dv(self, x):
        DV, T, arr_vinf, lamberts = self.decode_state_eval_fitness(x)
        return sum(DV)