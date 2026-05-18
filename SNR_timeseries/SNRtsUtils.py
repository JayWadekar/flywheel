#
#    Copyright (c) 2025 Francesco Iacovelli <francesco.iacovelli@unige.ch>
#
#    All rights reserved. Use of this source code is governed by the
#    license that can be found in the LICENSE file.

import numpy as np
import os
import ast
import h5py
import SNRtsGlobals as glob

##############################################################################
# LOAD
##############################################################################

def load_population(name, nEventsUse=None, keys_skip=[]):
    
    """
    Load a dictionary containing the events parameters in h5 file, compute some useful combinations and perform checks.
    
    :param str name: The name of the file to load the events from. This has to include the path and the ``h5`` or ``hdf5`` extension.
    :param int or None nEventsUse: Number of the events in the given file to load.
    :type kind: int or None
    :param list(str) calculate_params: Parameters not present in the file to compute. The supported parameters are ``'LambdaTilde'``, ``'deltaLambda'``, ``'Lambda1'``, ``'Lambda2'``, ``'theta'``, ``'phi'``, ``'ra'``, ``'dec'``.
    :param list(str) keys_skip: Parameters present in the file to skip.
    
    :return: Dictionary containing the loaded events, as in :py:data:`events`.
    :rtype: dict(array, array, ...)
    
    """
    
    events={}
    with h5py.File(name, 'r') as f:
        for key in f.keys():
            if key not in keys_skip:
                events[key] = np.array(f[key][:])
            else:
                print('Skipping %s' %key)
        if nEventsUse is not None:
            for key in f.keys():
                events[key]=events[key][:nEventsUse]
    
    return events

##############################################################################
# ANGLES
##############################################################################

# See http://spiff.rit.edu/classes/phys440/lectures/coords/coords.html
# Check: https://www.vercalendario.info/en/how/convert-ra-degrees-hours.html

def ra_dec_from_th_phi_rad(theta, phi):
    """
    Compute :math:`\\alpha` and :math:`\delta` in :math:`\\rm rad` from :math:`\\theta` and :math:`\phi` in :math:`\\rm rad`.
    
    :param array or float theta: The :math:`\\theta` sky position angle(s) to convert, in :math:`\\rm rad`.
    :param array or float phi: The :math:`\phi` sky position angle(s) to convert, in :math:`\\rm rad`.
    
    :return: :math:`\\alpha` and :math:`\delta` in :math:`\\rm rad`.
    :rtype: tuple(array, array) or tuple(float, float)
    
    """
    ra = phi #np.rad2deg(phi)
    dec = 0.5*np.pi - theta #np.rad2deg(0.5 * np.pi - theta)
    return ra, dec

def th_phi_from_ra_dec_rad(ra, dec):
    """
    Compute :math:`\\theta` and :math:`\phi` in :math:`\\rm rad` from :math:`\\alpha` and :math:`\delta` in :math:`\\rm rad`.
    
    :param array or float ra: The :math:`\\alpha` sky position angle(s) to convert, in :math:`\\rm rad`.
    :param array or float dec: The The :math:`\delta` sky position angle(s) angle(s) to convert, in :math:`\\rm rad`.
    
    :return: :math:`\\theta` and :math:`\phi` in :math:`\\rm rad`.
    :rtype: tuple(array, array) or tuple(float, float)
    
    """
    theta = 0.5 * np.pi - dec
    phi = ra
    return theta, phi


def ra_dec_from_th_phi(theta, phi):
    """
    Compute :math:`\\alpha` and :math:`\delta` in :math:`\\rm deg` from :math:`\\theta` and :math:`\phi` in :math:`\\rm rad`.
    
    :param array or float theta: The :math:`\\theta` sky position angle(s) to convert, in :math:`\\rm rad`.
    :param array or float phi: The :math:`\phi` sky position angle(s) to convert, in :math:`\\rm rad`.
    
    :return: :math:`\\alpha` and :math:`\delta` in :math:`\\rm deg`.
    :rtype: tuple(array, array) or tuple(float, float)
    
    """
    ra = np.rad2deg(phi)
    dec = np.rad2deg(0.5 * np.pi - theta)
    return ra, dec

def th_phi_from_ra_dec(ra, dec):
    """
    Compute :math:`\\theta` and :math:`\phi` in :math:`\\rm rad` from :math:`\\alpha` and :math:`\delta` in :math:`\\rm deg`.
    
    :param array or float ra: The :math:`\\alpha` sky position angle(s) to convert, in :math:`\\rm deg`.
    :param array or float dec: The The :math:`\delta` sky position angle(s) angle(s) to convert, in :math:`\\rm deg`.
    
    :return: :math:`\\theta` and :math:`\phi` in :math:`\\rm rad`.
    :rtype: tuple(array, array) or tuple(float, float)
    
    """
    theta = 0.5 * np.pi - np.deg2rad(dec)
    phi = np.deg2rad(ra)
    return theta, phi

def deg_min_sec_to_decimal_deg(d, m, s):
    """
    Convert one or multiple angles in degrees, minutes, seconds to decimal degrees.
    
    :param array or float d: The degrees of the angle(s) to convert.
    :param array or float m: The minutes of the angle(s) to convert.
    :param array or float s: The seconds of the angle(s) to convert.
    
    :return: The angle(s) in decimal degrees.
    :rtype: array or float
    
    """
    return d + m/60 + s/3600

def hr_min_sec_to_decimal_deg(h, m, s):
    """
    Convert one or multiple angles in hours, minutes, seconds to decimal degrees.
    
    :param array or float h: The hours of the angle(s) to convert.
    :param array or float m: The minutes of the angle(s) to convert.
    :param array or float s: The seconds of the angle(s) to convert.
    
    :return: The angle(s) in decimal degrees.
    :rtype: array or float
    
    """
    # decimal degrees=15*h+15*m/60+15*s/3600.
    
    return 15*(h+m/60+s/3600)

def deg_min_sec_to_rad(d, m, s):
    """
    Convert one or multiple angles in degrees, minutes, seconds to :math:`\\rm rad`.
    
    :param array or float d: The degrees of the angle(s) to convert.
    :param array or float m: The minutes of the angle(s) to convert.
    :param array or float s: The seconds of the angle(s) to convert.
    
    :return: The angle(s) in :math:`\\rm rad`.
    :rtype: array or float
    
    """
    return deg_min_sec_to_decimal_deg(d, m, s)*np.pi/180

def hr_min_sec_to_rad(h, m, s):
    """
    Convert one or multiple angles in hours, minutes, seconds to :math:`\\rm rad`.
    
    :param array or float h: The hours of the angle(s) to convert.
    :param array or float m: The minutes of the angle(s) to convert.
    :param array or float s: The seconds of the angle(s) to convert.
    
    :return: The angle(s) in :math:`\\rm rad`.
    :rtype: array or float
    
    """
    return hr_min_sec_to_decimal_deg(h, m, s)*np.pi/180


def rad_to_deg_min_sec(rad):
    """
    Convert one or multiple angles in :math:`\\rm rad` to degrees, minutes, seconds.
    
    Checks have been performed with `<https://www.calculatorsoup.com/calculators/conversions/convert-decimal-degrees-to-degrees-minutes-seconds.php>`_.
    
    :param array or float rad: The angle(s) in :math:`\\rm rad`.
    
    :return: The angle(s)' degrees, minutes, seconds.
    :rtype: tuple(array, array, array) or tuple(float, float, float)
    
    """
    # check: https://www.calculatorsoup.com/calculators/conversions/convert-decimal-degrees-to-degrees-minutes-seconds.php
    
    d = np.floor(rad).astype(int)
    
    m_exact = (rad-d)*60
    m = np.floor(m_exact).astype(int)

    s = np.round((m_exact - m)*60, 0).astype(int)
    
    return d, m, s

def rad_to_hr_min_sec(rad):
    """
    Convert one or multiple angles in :math:`\\rm rad` to hours, minutes, seconds.
    
    :param array or float rad: The angle(s) in :math:`\\rm rad`.
    
    :return: The angle(s)' hours, minutes, seconds.
    :rtype: tuple(array, array, array) or tuple(float, float, float)
    
    """
    hh = rad/15
    h = np.floor(hh).astype(int)
    
    m_exact = (hh-h)*60
    m = np.floor(m_exact).astype(int)

    s = np.round((m_exact - m)*60, 0).astype(int)
    
    return h, m, s

def hr_min_sec_string(h,m,s):
    """
    Convert one or multiple angles in hours, minutes, seconds to strings.
    
    :param array or float h: The hours of the angle(s) to convert.
    :param array or float m: The minutes of the angle(s) to convert.
    :param array or float s: The seconds of the angle(s) to convert.
    
    :return: The string(s) containing the angle(s).
    :rtype: list(str) or str
    
    """
    #h,m,s = np.asarray(h), np.asarray(m), np.asarray(s)
    #s = int(np.round(s,0))
    try:
        return [ str((h[i]))+'h'+str((m[i]))+'m'+str(s[i])+'s' for i in range(len(h))]
    except TypeError:
        return str((h))+'h'+str((m))+'m'+str(s)+'s'

def deg_min_sec_string(d,m,s):
    """
    Convert one or multiple angles in degrees, minutes, seconds to strings.
    
    :param array or float d: The degrees of the angle(s) to convert.
    :param array or float m: The minutes of the angle(s) to convert.
    :param array or float s: The seconds of the angle(s) to convert.
    
    :return: The string(s) containing the angle(s).
    :rtype: list(str) or str
    
    """
    #d,m,s = np.asarray(d), np.asarray(m), np.asarray(s)
    #s = int(s)
    
    try:
        return [ str((d[i]))+'°'+str((m[i]))+'m'+str(s[i])+'s' for i in range(len(d))]
    except TypeError:
        return  str((d))+'°'+str((m))+'m'+str(s)+'s'
    
def theta_to_dec_degminsec(theta):
    """
    Compute :math:`\\delta` in degree, minutes, seconds from :math:`\\theta`.
    
    :param array or float theta: The :math:`\\theta` sky position angle(s) to convert.
    
    :return: :math:`\\delta` in degree, minutes, seconds.
    :rtype: list(str) or str
    
    """
    dec = np.rad2deg(0.5 * np.pi - theta)
    return deg_min_sec_string(*rad_to_deg_min_sec(dec))

def phi_to_ra_hrms(phi):
    """
    Compute :math:`\\alpha` in hours, minutes, seconds from :math:`\phi`.
    
    :param array or float phi: The :math:`\phi` sky position angle(s) to convert.
    
    :return: :math:`\\alpha` in hours, minutes, seconds.
    :rtype: list(str) or str
    
    """
    ra = np.rad2deg(phi)
    return hr_min_sec_string(*rad_to_hr_min_sec(ra))

def phi_to_ra_degminsec(phi):
    """
    Compute :math:`\\alpha` in degree, minutes, seconds from :math:`\phi`.
    
    :param array or float phi: The :math:`\phi` sky position angle(s) to convert.
    
    :return: :math:`\\alpha` in degree, minutes, seconds.
    :rtype: list(str) or str
    
    """
    ra = np.rad2deg(phi)
    return deg_min_sec_string(*rad_to_deg_min_sec(ra)) #hr_min_sec_string(*rad_to_hr_min_sec(ra))

def gal_l_b_from_ra_dec(ra, dec):
    """
    Compute galactic longitude and latitude from right ascension and declination.

    :param array or float ra: The :math:`\\alpha` sky position angle(s) to convert, in :math:`\\rm deg`.
    :param array or float dec: The The :math:`\delta` sky position angle(s) angle(s) to convert, in :math:`\\rm deg`.
    
    :return: :math:`l` and :math:`b` in :math:`\\rm rad`.
    :rtype: tuple(array, array) or tuple(float, float)
    """
    
    dec_ngp = 0.4734772828041517
    ra_ngp  = 3.366033268750004
    l_ncp   = 2.145588052815142
    b = np.arcsin(np.sin(dec)*np.sin(dec_ngp) + np.cos(dec)*np.cos(ra-ra_ngp)*np.cos(dec_ngp))
    l = l_ncp - np.arctan2(np.sin(ra-ra_ngp)*np.cos(dec), np.sin(dec)*np.cos(dec_ngp) - np.cos(dec)*np.sin(dec_ngp)*np.cos(ra-ra_ngp))
    
    return l, b

def ra_dec_from_gal_l_b(l, b):
    """
    Compute right ascension and declination from galactic longitude and latitude.

    :param array or float l: The :math:`l` sky position angle(s) to convert, in :math:`\\rm rad`.
    :param array or float b: The The :math:`b` sky position angle(s) angle(s) to convert, in :math:`\\rm rad`.
    
    :return: :math:`\\alpha` and :math:`\delta` in :math:`\\rm rad`.
    :rtype: tuple(array, array) or tuple(float, float)
    """
    
    dec_ngp = 0.4734772828041517
    ra_ngp  = 3.366033268750004
    l_ncp   = 2.145588052815142
    dec = np.arcsin(np.sin(b)*np.sin(dec_ngp) + np.cos(b)*np.cos(l_ncp-l)*np.cos(dec_ngp))
    ra  = ra_ngp + np.arctan2(np.sin(l_ncp-l)*np.cos(b), np.sin(b)*np.cos(dec_ngp) - np.cos(b)*np.sin(dec_ngp)*np.cos(l_ncp-l)) 
    
    return np.mod(ra, 2.*np.pi), dec

##############################################################################
# TIDAL PARAMETERS
##############################################################################

def Lamt_delLam_from_Lam12(Lambda1, Lambda2, eta):
    """
    Compute the dimensionless tidal deformability combinations :math:`\\tilde{\Lambda}` and :math:`\delta\\tilde{\Lambda}`, defined in `arXiv:1402.5156 <https://arxiv.org/abs/1402.5156>`_ eq. (5) and (6), as a function of the dimensionless tidal deformabilities of the two objects and the symmetric mass ratio.
    
    :param array or float Lambda1: Tidal deformability of object 1, :math:`\Lambda_1`.
    :param array or float Lambda2: Tidal deformability of object 2, :math:`\Lambda_2`.
    :param array or float eta: The symmetric mass ratio(s), :math:`\eta`, of the objects.
    :return: :math:`\\tilde{\Lambda}` and :math:`\delta\\tilde{\Lambda}`.
    :rtype: tuple(array, array) or tuple(float, float)
    
    """
    eta2 = eta*eta
    # This is needed to stabilize JAX derivatives
    Seta = np.sqrt(np.where(eta<0.25, 1.0 - 4.0*eta, 0.))
        
    Lamt = (8./13.)*((1. + 7.*eta - 31.*eta2)*(Lambda1 + Lambda2) + Seta*(1. + 9.*eta - 11.*eta2)*(Lambda1 - Lambda2))
    
    delLam = 0.5*(Seta*(1. - 13272./1319.*eta + 8944./1319.*eta2)*(Lambda1 + Lambda2) + (1. - 15910./1319.*eta + 32850./1319.*eta2 + 3380./1319.*eta2*eta)*(Lambda1 - Lambda2))
    
    return Lamt, delLam
    
def Lam12_from_Lamt_delLam(Lamt, delLam, eta):
    """
    Compute the dimensionless tidal deformabilities of the two objects as a function of the dimensionless tidal deformability combinations :math:`\\tilde{\Lambda}` and :math:`\delta\\tilde{\Lambda}`, defined in `arXiv:1402.5156 <https://arxiv.org/abs/1402.5156>`_ eq. (5) and (6), and the symmetric mass ratio.
    
    :param array or float Lamt: Tidal deformability combination :math:`\\tilde{\Lambda}`.
    :param array or float delLam: Tidal deformability combination :math:`\delta\\tilde{\Lambda}`.
    :param array or float eta: The symmetric mass ratio(s), :math:`\eta`, of the objects.
    :return: :math:`\Lambda_1` and :math:`\Lambda_2`.
    :rtype: tuple(array, array) or tuple(float, float)
    
    """
        
    eta2 = eta*eta
    Seta = np.sqrt(np.where(eta<0.25, 1.0 - 4.0*eta, 0.))
    
    mLp=(8./13.)*(1.+ 7.*eta-31.*eta2)
    mLm=(8./13.)*Seta*(1.+ 9.*eta-11.*eta2)
    mdp=Seta*(1.-(13272./1319.)*eta+(8944./1319.)*eta2)*0.5
    mdm=(1.-(15910./1319.)*eta+(32850./1319.)*eta2+(3380./1319.)*(eta2*eta))*0.5

    det=(306656./1319.)*(eta**5)-(5936./1319.)*(eta**4)

    Lambda1 = ((mdp-mdm)*Lamt+(mLm-mLp)*delLam)/det
    Lambda2 = ((-mdm-mdp)*Lamt+(mLm+mLp)*delLam)/det
    
    return Lambda1, Lambda2

##############################################################################
# MASSES
##############################################################################

def m1m2_from_Mceta(Mc, eta):
    """
    Compute the component masses of a binary given its chirp mass and symmetric mass ratio.
    
    :param array or float Mc: Chirp mass of the binary, :math:`{\cal M}_c`.
    :param array or float eta: The symmetric mass ratio(s), :math:`\eta`, of the objects.
    :return: :math:`m_1` and :math:`m_2`.
    :rtype: tuple(array, array) or tuple(float, float)
    
    """
    Seta = np.sqrt(np.where(eta<0.25, 1.0 - 4.0*eta, 0.))
    m1 = 0.5*(Mc/(eta**(3./5.)))*(1. + Seta)
    m2 = 0.5*(Mc/(eta**(3./5.)))*(1. - Seta)

    return m1, m2
    
def Mceta_from_m1m2(m1, m2):
    """
    Compute the chirp mass and symmetric mass ratio of a binary given its component masses.
    
    :param array or float m1: Mass of the primary object, :math:`m_1`.
    :param array or float m2: Mass of the secondary object, :math:`m_2`.
    :return: :math:`{\cal M}_c` and :math:`\eta`.
    :rtype: tuple(array, array) or tuple(float, float)
    
    """
    Mc  = ((m1*m2)**(3./5.))/((m1+m2)**(1./5.))
    eta = (m1*m2)/((m1+m2)*(m1+m2))
    
    return Mc, eta


def m1m2_from_Mclogq(Mc, lq):
    """
    Compute the component masses of a binary given its chirp mass and log mass ratio.
    Mass ratio is defined as q = m2/m1
    """
    q = np.exp(lq)
    m1 = Mc*(1+q)**(1./5.)/q**(3./5.)
    m2 = q*m1

    return m1, m2

def Mc_eta_from_Mtot_q(Mtot, q):
    """
    Compute the chirp mass and symmetric mass ratio of a binary given its total mass and mass ratio.
    
    :param array or float Mtot: Total mass of the binary, :math:`M_{\\rm tot}`.
    :param array or float q: Mass ratio of the binary, :math:`q>1`.
    :return: :math:`{\cal M}_c` and :math:`\eta`.
    :rtype: tuple(array, array) or tuple(float, float)
    
    """
    
    eta = q/(1.+q)**2
    Mc = Mtot*(eta)**(3/5)
    return Mc, eta

def Mtot_q_from_Mc_eta(Mc, eta):
    """
    Compute the total mass and mass ratio of a binary given its chirp mass and symmetric mass ratio.

    :param array or float Mc: Chirp mass of the binary, :math:`{\cal M}_c`.
    :param array or float eta: Symmetric mass ratio of the binary, :math:`\eta`.
    :return: :math:`M_{\\rm tot}` and :math:`q`.
    :rtype: tuple(array, array) or tuple(float, float)
    """
    
    Seta = np.sqrt(np.where(eta<0.25, 1.0 - 4.0*eta, 0.))
    q = (1. + Seta) / (1. - Seta)
    Mtot = Mc*eta**(-3/5)
    return Mtot, q

##############################################################################
# SPINS
##############################################################################

def zrot(angle, vx, vy, vz):
    """
    Perform a rotation of the components of a vector around the :math:`z` axis by a given angle.
    
    :param array or float angle: Rotation angle(s).
    :param array or float vx: The :math:`x` component(s) of the vector(s).
    :param array or float vy: The :math:`y` component(s) of the vector(s).
    :param array or float vz: The :math:`z` component(s) of the vector(s).
    
    :return: The components of the rotated vector(s) around :math:`z`.
    :rtype: tuple(array, array, array) or tuple(float, float, float)
    
    """
    # Function to perform a rotation of the components of a vector around the z axis by a given angle
    tmp = vx*np.cos(angle) - vy*np.sin(angle)
    yy  = vx*np.sin(angle) + vy*np.cos(angle)
    xx  = tmp
    return xx, yy, vz

def yrot(angle, vx, vy, vz):
    """
    Perform a rotation of the components of a vector around the :math:`y` axis by a given angle.
    
    :param array or float angle: Rotation angle(s).
    :param array or float vx: The :math:`x` component(s) of the vector(s).
    :param array or float vy: The :math:`y` component(s) of the vector(s).
    :param array or float vz: The :math:`z` component(s) of the vector(s).
    
    :return: The components of the rotated vector(s) around :math:`y`.
    :rtype: tuple(array, array, array) or tuple(float, float, float)
    
    """
    # Function to perform a rotation of the components of a vector around the y axis by a given angle
    tmp = vx*np.cos(angle) + vz*np.sin(angle)
    zz  = - vx*np.sin(angle) + vz*np.cos(angle)
    xx  = tmp
    return xx, vy, zz

def TransformPrecessing_angles2comp(thetaJN, phiJL, theta1, theta2, phi12, chi1, chi2, Mc, eta, fRef, phiRef):
    """
    Compute the components of the spin in cartesian frame given the angular variables.
    Adapted from :py:class:`LALSimInspiral.c`, function :py:class:`XLALSimInspiralTransformPrecessingNewInitialConditions`, line 5885.
    For a scheme of the conventions, see `<https://lscsoft.docs.ligo.org/lalsuite/lalsimulation/group__lalsimulation__inference.html>`_.
    
    :param array or float thetaJN: Inclination between total angular momentum (:math:`J`) and the direction of propagation, :math:`\\theta_{JN}` (so that :math:`\\theta_{JN} \\to \iota` for :math:`\\chi_1 + \\chi_2 \\to 0`).
    :param array or float phiJL: Azimuthal angle of the Newtonian orbital angular momentum :math:`L_N` on its cone about the total angular momentum :math:`J`, :math:`\phi_{JL}`.
    :param array or float theta1: Inclination (tilt angle) of object 1 measured from the Newtonian orbital angular momentum (:math:`L_N`), :math:`\\theta_{s,1}`.
    :param array or float theta2: Inclination (tilt angle) of object 2 measured from the Newtonian orbital angular momentum (:math:`L_N`), :math:`\\theta_{s,2}`.
    :param array or float phi12: Difference in azimuthal angles between the two spins, :math:`\phi_{1,2}`.
    :param array or float chi1: Dimensionless spin magnitude of object 1, :math:`\chi_1`.
    :param array or float chi2: Dimensionless spin magnitude of object 2, :math:`\chi_2`.
    :param array or float Mc: Chirp mass of the binary, :math:`{\cal M}_c`, in units of :math:`\\rm M_{\odot}`.
    :param array or float eta: The symmetric mass ratio(s), :math:`\eta`, of the objects.
    :param array or float fRef: Reference frequency, in :math:`\\rm Hz`.
    :param array or float phiRef: Reference phase, in :math:`\\rm rad`.
    
    :return: :math:`\iota`, :math:`\chi_{1,x}`, :math:`\chi_{1,y}`, :math:`\chi_{1,z}`, :math:`\chi_{2,x}`, :math:`\chi_{2,y}`, :math:`\chi_{2,z}`.
    :rtype: tuple(array, array, array, array, array, array, array) or tuple(float, float, float, float, float, float, float)
    
    """
    
    LNhx = 0.
    LNhy = 0.
    LNhz = 1.

    s1hatx = np.sin(theta1) * np.cos(phiRef)
    s1haty = np.sin(theta1) * np.sin(phiRef)
    s1hatz = np.cos(theta1)
    s2hatx = np.sin(theta2) * np.cos(phi12+phiRef)
    s2haty = np.sin(theta2) * np.sin(phi12+phiRef)
    s2hatz = np.cos(theta2)

    m1, m2 = m1m2_from_Mceta(Mc, eta)
    M = m1+m2
    v0 = (M * glob.GMsun_over_c3 * np.pi * fRef)**(1./3.)

    # Define S1, S2, J with proper magnitudes
    Lmag = (M*M*eta/v0)*(1. + v0*v0*(1.5 + eta/6.))
    
    s1x = m1 * m1 * chi1 * s1hatx
    s1y = m1 * m1 * chi1 * s1haty
    s1z = m1 * m1 * chi1 * s1hatz
    s2x = m2 * m2 * chi2 * s2hatx
    s2y = m2 * m2 * chi2 * s2haty
    s2z = m2 * m2 * chi2 * s2hatz
    Jx = s1x + s2x
    Jy = s1y + s2y
    Jz = Lmag + s1z + s2z

    # Normalize J to Jhat, find its angles in starting frame

    Jnorm = np.sqrt(Jx*Jx + Jy*Jy + Jz*Jz)
    Jhatx = Jx / Jnorm
    Jhaty = Jy / Jnorm
    Jhatz = Jz / Jnorm
    theta0 = np.arccos(Jhatz)
    phi0 = np.arctan2(np.real(Jhaty), np.real(Jhatx))
    
    # Rotation 1: Rotate about z-axis by -phi0 to put Jhat in x-z plane
    s1hatx, s1haty, s1hatz = zrot(-phi0, s1hatx, s1haty, s1hatz)
    s2hatx, s2haty, s2hatz = zrot(-phi0, s2hatx, s2haty, s2hatz)

    # Rotation 2: Rotate about new y-axis by -theta0 to put Jhat along z-axis
    LNhx, LNhy, LNhz       = yrot(-theta0, LNhx, LNhy, LNhz)
    s1hatx, s1haty, s1hatz = yrot(-theta0, s1hatx, s1haty, s1hatz)
    s2hatx, s2haty, s2hatz = yrot(-theta0, s2hatx, s2haty, s2hatz)

    # Rotation 3: Rotate about new z-axis by phiJL to put L at desired azimuth about J.
    # Note that is currently in x-z plane towards -x (i.e. azimuth=pi). Hence we rotate about z by phiJL - pi
    LNhx, LNhy, LNhz       = zrot(phiJL - np.pi, LNhx, LNhy, LNhz)
    s1hatx, s1haty, s1hatz = zrot(phiJL - np.pi, s1hatx, s1haty, s1hatz)
    s2hatx, s2haty, s2hatz = zrot(phiJL - np.pi, s2hatx, s2haty, s2hatz)
    
    # The cosine of the angle between L and N is the scalar product of the two vectors, no further rotation needed
    
    Nx=0.
    Ny=np.sin(thetaJN)
    Nz=np.cos(thetaJN)
    iota=np.arccos(Nx*LNhx+Ny*LNhy+Nz*LNhz)

    # Rotation 4-5: Now J is along z and N in y-z plane, inclined from J by thetaJN and with >ve component along y.
    # Now we bring L into the z axis to get spin components.
    thetaLJ = np.arccos(LNhz)
    phiL    = np.arctan2(np.real(LNhy), np.real(LNhx))
    
    s1hatx, s1haty, s1hatz = zrot(-phiL, s1hatx, s1haty, s1hatz)
    s2hatx, s2haty, s2hatz = zrot(-phiL, s2hatx, s2haty, s2hatz)
    Nx, Ny, Nz             = zrot(-phiL, Nx, Ny, Nz)
    
    s1hatx, s1haty, s1hatz = yrot(-thetaLJ, s1hatx, s1haty, s1hatz)
    s2hatx, s2haty, s2hatz = yrot(-thetaLJ, s2hatx, s2haty, s2hatz)
    Nx, Ny, Nz             = yrot(-thetaLJ, Nx, Ny, Nz)
    
    # Rotation 6: Now L is along z and we have to bring N in the y-z plane with >ve y components.
    
    phiN = np.arctan2(np.real(Ny), np.real(Nx))
    
    s1hatx, s1haty, s1hatz = zrot(np.pi/2.-phiN-phiRef, s1hatx, s1haty, s1hatz)
    s2hatx, s2haty, s2hatz = zrot(np.pi/2.-phiN-phiRef, s2hatx, s2haty, s2hatz)
    
    S1x = s1hatx*chi1
    S1y = s1haty*chi1
    S1z = s1hatz*chi1
    S2x = s2hatx*chi2
    S2y = s2haty*chi2
    S2z = s2hatz*chi2
    
    return iota, S1x, S1y, S1z, S2x, S2y, S2z

def TransformPrecessing_comp2angles(iota, S1x, S1y, S1z, S2x, S2y, S2z, Mc, eta, fRef, phiRef):
    """
    Compute the angular variables of the spins given the components in cartesian frame
    Adapted from :py:class:`LALSimInspiral.c`, function :py:class:`XLALSimInspiralTransformPrecessingWvf2PE`, line 6105.
    For a scheme of the conventions, see `<https://lscsoft.docs.ligo.org/lalsuite/lalsimulation/group__lalsimulation__inference.html>`_.
    
    :param array or float iota: Inclination between the orbital angular momentum and the direction of propagation.
    :param array or float S1x: spin of object 1 along the axis :math:`x`, :math:`\chi_{1,x}`.
    :param array or float S1y: spin of object 1 along the axis :math:`y`, :math:`\chi_{1,y}`.
    :param array or float S1z: spin of object 1 along the axis :math:`z`, :math:`\chi_{1,z}`.
    :param array or float S2x: spin of object 2 along the axis :math:`x`, :math:`\chi_{2,x}`.
    :param array or float S2y: spin of object 2 along the axis :math:`y`, :math:`\chi_{2,y}`.
    :param array or float S2z: spin of object 2 along the axis :math:`z`, :math:`\chi_{2,z}`.
    :param array or float Mc: Chirp mass of the binary, :math:`{\cal M}_c`, in units of :math:`\\rm M_{\odot}`.
    :param array or float eta: The symmetric mass ratio(s), :math:`\eta`, of the objects.
    :param array or float fRef: Reference frequency, in :math:`\\rm Hz`.
    :param array or float phiRef: Reference phase, in :math:`\\rm rad`.
    
    :return: :math:`\\theta_{JN}`, :math:`\phi_{JL}`, :math:`\\theta_{s,1}`, :math:`\\theta_{s,2}`, :math:`\phi_{1,2}`, :math:`\chi_1`, :math:`\chi_2`.
    :rtype: tuple(array, array, array, array, array, array, array) or tuple(float, float, float, float, float, float, float)
    
    """
    
    LNhx = 0.
    LNhy = 0.
    LNhz = 1.
    chi1 = np.sqrt(S1x*S1x + S1y*S1y + S1z*S1z)
    chi2 = np.sqrt(S2x*S2x + S2y*S2y + S2z*S2z)
    
    s1hatx = np.where(chi1>0., S1x/(chi1), 0.)
    s1haty = np.where(chi1>0., S1y/(chi1), 0.)
    s1hatz = np.where(chi1>0., S1z/(chi1), 0.)
    s2hatx = np.where(chi2>0., S2x/(chi2), 0.)
    s2haty = np.where(chi2>0., S2y/(chi2), 0.)
    s2hatz = np.where(chi2>0., S2z/(chi2), 0.)
    
    phi1 = np.arctan2(np.real(s1haty), np.real(s1hatx))
    phi2 = np.arctan2(np.real(s2haty), np.real(s2hatx))
    
    phi12 = np.where(phi2 - phi1 < 0., 2.*np.pi + (phi2 - phi1), phi2 - phi1)
    
    theta1 = np.arccos(s1hatz)
    theta2 = np.arccos(s2hatz)
    
    m1, m2 = m1m2_from_Mceta(Mc, eta)
    M = m1+m2
    v0 = (M * glob.GMsun_over_c3 * np.pi * fRef)**(1./3.)#np.cbrt(M * glob.GMsun_over_c3 * np.pi * fRef)
    # Define S1, S2, J with proper magnitudes
    Lmag = (M*M*eta/v0)*(1. + v0*v0*(1.5 + eta/6.))
    
    s1x = m1 * m1 * S1x
    s1y = m1 * m1 * S1y
    s1z = m1 * m1 * S1z
    s2x = m2 * m2 * S2x
    s2y = m2 * m2 * S2y
    s2z = m2 * m2 * S2z
    Jx = s1x + s2x
    Jy = s1y + s2y
    Jz = Lmag*LNhz + s1z + s2z
    
    # Normalize J to Jhat, find its angles in starting frame
    
    Jnorm = np.sqrt(Jx*Jx + Jy*Jy + Jz*Jz)
    Jhatx = Jx / Jnorm
    Jhaty = Jy / Jnorm
    Jhatz = Jz / Jnorm
    thetaJL = np.arccos(Jhatz)
    phiJ    = np.arctan2(np.real(Jhaty), np.real(Jhatx))
    
    phiO = np.pi/2. - phiRef
    Nx = np.sin(iota)*np.cos(phiO)
    Ny = np.sin(iota)*np.sin(phiO)
    Nz = np.cos(iota)
    
    thetaJN = np.arccos(Jhatx*Nx + Jhaty*Ny + Jhatz*Nz)
    
    # The easiest way to define the phiJL is to rotate to the frame where J is along z and N is in the y-z plane
    Nx, Ny, Nz = zrot(-phiJ, Nx, Ny, Nz)
    Nx, Ny, Nz = yrot(-thetaJL, Nx, Ny, Nz)
    
    LNhx, LNhy, LNhz = zrot(-phiJ, LNhx, LNhy, LNhz)
    LNhx, LNhy, LNhz = yrot(-thetaJL, LNhx, LNhy, LNhz)
    
    phiN = np.arctan2(np.real(Ny), np.real(Nx))
    
    # After rotation defined below N should be in y-z plane inclined by thetaJN to J=z
    LNhx, LNhy, LNhz = zrot(np.pi/2. - phiN, LNhx, LNhy, LNhz)
    
    phiJL = np.arctan2(np.real(LNhy), np.real(LNhx))
    phiJL = np.where(phiJL<0., phiJL+2.*np.pi, phiJL)
    
    return thetaJN, phiJL, theta1, theta2, phi12, chi1, chi2
    
##############################################################################
# TIMES
##############################################################################

def GPSt_to_J200t(t_GPS):
    # According to https://www.andrews.edu/~tzs/timeconv/timedisplay.php the GPS time of J2000 is 630763148 s
    return t_GPS - 630763148.0

def GPSt_to_LMST(t_GPS, lat, long):
    """
    Compute the Local Mean Sidereal Time (LMST) in units of fraction of day, from GPS time and location (given as latitude and longitude in degrees)
    
    :param array or float t_GPS: GPS time(s) to convert, in seconds.
    :param float lat: Latitude of the chosen location, in :math:`\\rm deg`.
    :param float long: Longitude of the chosen location, in :math:`\\rm deg`.
    
    :return: Local Mean Sidereal Time(s).
    :rtype: array or float
    
    """
    from astropy.coordinates import EarthLocation
    import astropy.time as aspyt
    import astropy.units as u
    # Uncomment the next two lines in case of troubles with IERS
    #import astropy
    #astropy.utils.iers.conf.iers_degraded_accuracy='ignore'

    loc = EarthLocation(lat=lat*u.deg, lon=long*u.deg)
    t = aspyt.Time(t_GPS, format='gps', location=(loc))
    LMST = t.sidereal_time('mean').value
    return LMST/24.

def GPSt_to_GMST_alt(t_GPS):
    """
    Compute the Greenwich Mean Sidereal Time (GMST) in units of fraction of day, from GPS time. This function does not rely on external libraries but is **approximate**.
    The implementation is taken from `GWFish <https://github.com/janosch314/GWFish/tree/main>`_.
    
    :param array or float t_GPS: GPS time(s) to convert, in seconds.
    
    :return: Greenwich Mean Sidereal Time(s).
    :rtype: array or float
    
    """
    
    return np.mod(9.533088395981618 + (t_GPS - 1126260000.) / 3600. * 24. / glob.siderealDay, 24.) / 24.
    
def GMST_to_LMST(tGMST, lat, long):
    """
    Compute the Local Mean Sidereal Time (LMST) in units of fraction of day, from Greenwich Mean Sidereal Time (GMST) and position given as latitude and longitude (in :math:`\\rm deg`)
    
    :param array or float tGMST: GMST to convert, in days.
    :param float lat: Latitude of the chosen location, in :math:`\\rm deg`.
    :param float long: Longitude of the chosen location, in :math:`\\rm deg`.
    
    :return: Local Mean Sidereal Time(s).
    :rtype: array or float
    
    """
    return np.remainder(tGMST.real + 1./15.*long/24., 1.)

def LMST_to_GMST(tLMST, lat, long):
    """
    Compute the Greenwich Mean Sidereal Time (GMST) in units of fraction of day, from Local Mean Sidereal Time (LMST) and position given as latitude and longitude (in :math:`\\rm deg`)
    
    :param array or float tLMST: LMST to convert, in days.
    :param float lat: Latitude of the chosen location, in :math:`\\rm deg`.
    :param float long: Longitude of the chosen location, in :math:`\\rm deg`.
    
    :return: Greenwich Mean Sidereal Time(s).
    :rtype: array or float
    
    """
    return np.remainder(tLMST.real - 1./15.*long/24., 1.)
    
def LunarMeanSiderealTime(t_GPS):
    """
    Compute the Lunar Mean Sidereal Time (LnMST) in units of fraction of lunar day, from GPS time
    
    :param array or float t_GPS: GPS time(s) to convert, in seconds.
    
    :return: Greenwich Mean Sidereal Time(s).
    :rtype: array or float
    
    """
    return np.remainder((t_GPS - 1126260000.) / glob.SiderealPeriodMoon, 1.)

def f_of_tau_star(t_to_coal, Mc, l, m):
    """
    Compute the frequency (in :math:`\\rm Hz`) at a specific time to coalescence (in seconds), given the events parameters.
    
    We use the expression in M. Maggiore - Gravitational Waves Vol. 1 eq. (4.20), valid in Newtonian and restricted PN approximation.
    
    :param array t_to_coal: Time to coalescence for which the frequency will be computed, in :math:`\\rm s`.
    :param dict(array, array, ...) kwargs: Dictionary with arrays containing the parameters of the events to compute the time to coalescence of, as in :py:data:`events`.
    :return: frequency evaluated at the desired times to coalescence for the chosen events evaluated, in hertz.
    :rtype: array
    
    """

    return m / (2. * np.pi) * (5. / 256. / t_to_coal)**(3./8.) * (glob.GMsun_over_c3 * Mc)**(-5./8.)

def tau_star_of_f(f, Mc, l, m):
    """
    Compute the time to coalescence (in seconds) at a specific frequency (in :math:`\\rm Hz`), given the events parameters.
    
    We use the expression in M. Maggiore - Gravitational Waves Vol. 1 eq. (4.20), valid in Newtonian and restricted PN approximation.
    
    :param array f: Frequency for which the time to coalescence will be computed, in :math:`\\rm Hz`.
    :param dict(array, array, ...) kwargs: Dictionary with arrays containing the parameters of the events to compute the time to coalescence of, as in :py:data:`events`.
    :return: Time to coalescence evaluated at the desired frequencies for the chosen events evaluated, in seconds.
    :rtype: array
    
    """

    return 5. / 256. * (glob.GMsun_over_c3 * Mc)**(-5./3.) / (2. * np.pi * f / m)**(-8./3.)
    
##############################################################################
# DETECTOR RELATIVE ORIENTATION AND DISTANCE
##############################################################################

def ang_btw_dets_GC(det1, det2):
    """
    Compute the angle between two detectors with respect to the great circle that joins them, see `<https://en.wikipedia.org/wiki/Great-circle_navigation>`_.
    
    :param dict(float, float, float) det1: Dictionary containing the latitude, ``'lat'``, longitude, ``'long'``, and orientation, ``'xax'``, of the first detector (all in degrees), as in :py:data:`gwfast.gwfastGlobals.detectors`.
    :param dict(float, float, float) det2: Dictionary containing the latitude, ``'lat'``, longitude, ``'long'``, and orientation, ``'xax'``, of the second detector (all in degrees), as in :py:data:`gwfast.gwfastGlobals.detectors`.
    
    :return: Angle between the two detectors, in :math:`\\rm deg`.
    :rtype: float
    
    """
    lat1, lat2   = np.deg2rad(det1['lat']), np.deg2rad(det2['lat'])
    long1, long2 = np.deg2rad(det1['long']), np.deg2rad(det2['long'])
    
    def initial_course(lat1, lat2, long1, long2):
        # Compute the course at the initial point given two points
        # See http://www.edwilliams.org/avform147.htm#Crs or https://en.wikipedia.org/wiki/Great-circle_navigation
        a = np.sin(long2-long1)*np.cos(lat2)
        b = np.cos(lat1)*np.sin(lat2)-np.sin(lat1)*np.cos(lat2)*np.cos(long2-long1)

        # If the initial point is a pole we need a "fix"
        return np.rad2deg(np.where(np.isclose(np.cos(lat1), 0.), np.where(lat1 > 0., np.pi, 2.*np.pi), np.arctan2(a,b)))
    
    def final_course(lat1, lat2, long1, long2):
        # Compute the course at the final point given two points
        # See http://www.edwilliams.org/avform147.htm#Crs or https://en.wikipedia.org/wiki/Great-circle_navigation
        a = np.sin(long2-long1)*np.cos(lat1)
        b = -np.cos(lat2)*np.sin(lat1)+np.sin(lat2)*np.cos(lat1)*np.cos(long2-long1)

        # If the final point is a pole we need a "fix"
        return np.rad2deg(np.where(np.isclose(np.cos(lat2), 0.), np.where(lat2 > 0., np.pi, 2.*np.pi), np.arctan2(a,b)))
    
    # Compute the course at the first detector
    ang1 = initial_course(lat1, lat2, long1, long2)
    # Compute the course at the second detector
    ang2 = final_course(lat1, lat2, long1, long2)

    angdiff = 360.-(ang2-ang1)

    return (det1['xax'] - det2['xax']) + np.where(angdiff<180.,angdiff, angdiff-360.)

def dist_btw_dets_GC(det1, det2):
    """
    Compute the great circle distance between two detectors using the Vincenty formula in spherical case, see `<https://en.wikipedia.org/wiki/Great-circle_distance>`_.
    
    :param dict(float, float, float) det1: Dictionary containing the latitude, ``'lat'``, longitude, ``'long'``, and orientation, ``'xax'``, of the first detector (all in degrees), as in :py:data:`gwfast.gwfastGlobals.detectors`.
    :param dict(float, float, float) det2: Dictionary containing the latitude, ``'lat'``, longitude, ``'long'``, and orientation, ``'xax'``, of the second detector (all in degrees), as in :py:data:`gwfast.gwfastGlobals.detectors`.
    
    :return: Great circle distance between the detectors, in :math:`\\rm km`.
    :rtype: float
    
    """

    lat1, lat2   = np.deg2rad(det1['lat']), np.deg2rad(det2['lat'])
    long1, long2 = np.deg2rad(det1['long']), np.deg2rad(det2['long'])
    dlong = long2 - long1

    num = np.sqrt((np.cos(lat2)*np.sin(dlong))**2 + (np.cos(lat1)*np.sin(lat2) - np.sin(lat1)*np.cos(lat2)*np.cos(dlong))**2)
    den = np.sin(lat1)*np.sin(lat2) + np.cos(lat1)*np.cos(lat2)*np.cos(dlong)

    return glob.REarth*np.arctan2(num, den)

def dist_btw_dets_Chord(det1, det2):
    """
    Compute the great circle chord length between two detectors, see `<https://en.wikipedia.org/wiki/Great-circle_distance>`_.
    
    :param dict(float, float, float) det1: Dictionary containing the latitude, ``'lat'``, longitude, ``'long'``, and orientation, ``'xax'``, of the first detector (all in degrees), as in :py:data:`gwfast.gwfastGlobals.detectors`.
    :param dict(float, float, float) det2: Dictionary containing the latitude, ``'lat'``, longitude, ``'long'``, and orientation, ``'xax'``, of the second detector (all in degrees), as in :py:data:`gwfast.gwfastGlobals.detectors`.
    
    :return: Great circle chord length between the detectors, in :math:`\\rm km`.
    :rtype: float
    
    """
    
    lat1, lat2   = np.deg2rad(det1['lat']), np.deg2rad(det2['lat'])
    long1, long2 = np.deg2rad(det1['long']), np.deg2rad(det2['long'])

    dx = np.cos(lat2)*np.cos(long2) - np.cos(lat1)*np.cos(long1)
    dy = np.cos(lat2)*np.sin(long2) - np.cos(lat1)*np.sin(long1)
    dz = np.sin(lat2) - np.sin(lat1)

    return glob.REarth*np.sqrt(dx*dx + dy*dy + dz*dz)

def bearing_angle(pt1, pt2): # This is the relative angle between the two detectors with respect to the North
    """
    Compute the bearing angle (i.e. the relative angle between the two points with respect to the North, clockwise) between two points, see `<http://www.movable-type.co.uk/scripts/latlong.html>`_.
    
    :param dict(float, float) pt1: Dictionary containing the latitude, ``'lat'`` and longitude, ``'long'`` of the first point (all in degrees).
    :param dict(float, float) pt2: Dictionary containing the latitude, ``'lat'`` and longitude, ``'long'`` of the second point (all in degrees).
    
    :return: Bearing angle between the points, in :math:`\\rm deg`.
    :rtype: float
    
    """
    lat1, lat2   = np.deg2rad(pt1['lat']), np.deg2rad(pt2['lat'])
    long1, long2 = np.deg2rad(pt1['long']), np.deg2rad(pt2['long'])

    X = np.cos(lat2) * np.sin(long2 - long1)
    Y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(long2 - long1)

    return np.rad2deg(np.arctan2(X, Y))

def xax_given_xarmpos(BS, vertex, angbtwArms):
    """
    Compute the angle between the bisector of a detector’s arms and local East, in degrees, given the position of the beam splitter and the vertex and the angle between the arms of the detector.
    
    :param dict(float, float) BS: Dictionary containing the latitude, ``'lat'`` and longitude, ``'long'`` of the beam splitter (all in degrees).
    :param dict(float, float) vertex: Dictionary containing the latitude, ``'lat'`` and longitude, ``'long'`` of the vertex (all in degrees).
    :param float angbtwArms: Angle between the arms of the detector, in degrees.
    :return: Detector orientation, in :math:`\\rm deg`.
    :rtype: float
    
    """

    return 90. - bearing_angle(BS, vertex) + angbtwArms/2.

def xvertexpos_given_bs_xax_length(BS, xax, armLength, angbtwArms):
    """
    Compute the position of the x-arm vertex of a detector, in :math:`\\rm (lat, long)`, given the position of the beam splitter, the angle between the bisector of a detector’s arms and local East, the arm length and the angle between the arms. See `<http://www.movable-type.co.uk/scripts/latlong.html>`_
    
    :param dict(float, float) BS: Dictionary containing the latitude, ``'lat'`` and longitude, ``'long'`` of the beam splitter (all in degrees).
    :param float xax: Detector orientation, in degrees.
    :param float armLength: Length of the arms of the detector, in :math:`\\rm km`.
    :param float angbtwArms: Angle between the arms of the detector, in degrees.
    
    :return: Position of the x-arm vertex of the detector, in :math:`\\rm (lat, long)`.
    :rtype: dict(float, float)
    
    """

    lat1, long1 = np.deg2rad(BS['lat']), np.deg2rad(BS['long'])
    bearing = np.deg2rad(90. - xax + angbtwArms/2.)
    delta = armLength/glob.REarth
    lat2 = np.arcsin(np.sin(lat1)*np.cos(delta) + np.cos(lat1)*np.sin(delta)*np.cos(bearing))
    long2 = long1 + np.arctan2(np.sin(bearing)*np.sin(delta)*np.cos(lat1), np.cos(delta)-np.sin(lat1)*np.sin(lat2))
    return {'lat':np.rad2deg(lat2), 'long':np.rad2deg(long2)}

def xyverticespos_given_bs_xax_length(BS, xax, armLength, angbtwArms):
    """
    Compute the position of the x-arm and y-arm vertices of a detector, in :math:`\\rm (lat, long)`, given the position of the beam splitter, the angle between the bisector of a detector’s arms and local East, the arm length and the angle between the arms. See `<http://www.movable-type.co.uk/scripts/latlong.html>`_
    
    :param dict(float, float) BS: Dictionary containing the latitude, ``'lat'`` and longitude, ``'long'`` of the beam splitter (all in degrees).
    :param float xax: Detector orientation, in degrees.
    :param float armLength: Length of the arms of the detector, in :math:`\\rm km`.
    :param float angbtwArms: Angle between the arms of the detector, in degrees.
    
    :return: Position of the x-arm vertex of the detector, in :math:`\\rm (lat, long)`.
    :rtype: dict(dict(float, float), dict(float, float))
    
    """

    lat1, long1 = np.deg2rad(BS['lat']), np.deg2rad(BS['long'])
    bearing1 = np.deg2rad(90. - xax + angbtwArms/2.)
    bearing2 = np.deg2rad(90. - xax + angbtwArms/2. + angbtwArms)
    delta = armLength/glob.REarth
    def latlong_given_bearing(bearing):
        lat = np.arcsin(np.sin(lat1)*np.cos(delta) + np.cos(lat1)*np.sin(delta)*np.cos(bearing))
        long = long1 + np.arctan2(np.sin(bearing)*np.sin(delta)*np.cos(lat1), np.cos(delta)-np.sin(lat1)*np.sin(lat))
        return lat, long
    lat2, long2 = latlong_given_bearing(bearing1)
    lat3, long3 = latlong_given_bearing(bearing2)
    return {'xarm':{'lat':np.rad2deg(lat2), 'long':np.rad2deg(long2)}, 'yarm':{'lat':np.rad2deg(lat3), 'long':np.rad2deg(long3)}}

def Earth_latlong_to_cartesian(lat_, long_, elevation, is_rad):
    
    if not is_rad:
        lat = np.deg2rad(lat_)
        long = np.deg2rad(long_)
    else:
        lat = lat_
        long = long_
        
    semi_major_axis = glob.SMajEarth  # for ellipsoid model of Earth, in km
    semi_minor_axis = glob.SMinEarth  # in km
    radius = semi_major_axis**2 / np.sqrt(semi_major_axis**2 * np.cos(lat)**2 + semi_minor_axis**2 * np.sin(lat)**2)

    x = (radius + elevation) * np.cos(lat) * np.cos(long)
    y = (radius + elevation) * np.cos(lat) * np.sin(long)
    z = ((semi_minor_axis / semi_major_axis)**2 * radius + elevation) * np.sin(lat)
    
    return x, y, z # in km

def arrival_time_difference(det1, det2, tGPS, ra, dec, is_rad=False):
    """
    Compute the arrival time difference between two detectors.
    
    Parameters
    ----------
    det1 : dict
        First detector dictionary with 'lat' and 'long' keys.
    det2 : dict
        Second detector dictionary with 'lat' and 'long' keys.
    tGPS : float
        GPS time in seconds.
    ra : float
        Right Ascension in radians.
    dec : float
        Declination in radians.
    
    Returns
    -------
    float
        Arrival time difference in seconds.
    """
    GMST = GPSt_to_LMST(tGPS, 0., 0.)*2*np.pi - ra
    
    x1, y1, z1 = Earth_latlong_to_cartesian(det1['lat'], det1['long'], det1['elevation'], is_rad=is_rad)
    x2, y2, z2 = Earth_latlong_to_cartesian(det2['lat'], det2['long'], det2['elevation'], is_rad=is_rad)

    res = (x2 - x1) * np.cos(dec) * np.cos(GMST) + (y2 - y1) * np.cos(dec) * np.sin(GMST) + (z2 - z1) * np.sin(dec)
    return res / glob.clight  # in seconds

def time_delay_from_geocenter(det1, tGPS, ra, dec, is_rad=False):
    """
    Compute the time delay from the geocenter to a detector.
    
    Parameters
    ----------
    det1 : dict
        Detector dictionary with 'lat' and 'long' keys.
    tGPS : float
        GPS time in seconds.
    ra : float
        Right Ascension in radians.
    dec : float
        Declination in radians.
    
    Returns
    -------
    float
        Time delay in seconds.
    """
    GMST = GPSt_to_LMST(tGPS, 0., 0.)*2*np.pi - ra
    x1, y1, z1 = Earth_latlong_to_cartesian(det1['lat'], det1['long'], det1['elevation'], is_rad=is_rad)
    res = x1 * np.cos(dec) * np.cos(GMST) + y1 * np.cos(dec) * np.sin(GMST) + z1 * np.sin(dec)
    return res / glob.clight  # in seconds

##############################################################################
# MERGER FREQUENCY
##############################################################################

def merger_frequency(evParams):
    """
    Compute the merger frequency for a binary black hole merger.
    
    Parameters
    ----------
    evParams : dict
        Dictionary containing the parameters of the event.
    
    Returns
    -------
    float
        Merger frequency in Hz.
    """
    def finalspin(eta, chi1, chi2):
        """
        Compute the spin of the final object, as in LALSimIMRPhenomD_internals.c line 161 and 142, which is taken from `arXiv:1508.07250 <https://arxiv.org/abs/1508.07250>`_ eq. (3.6).
        
        :param array or float eta: Symmetric mass ratio of the objects.
        :param array or float chi1: Spin of the primary object.
        :param array or float chi2: Spin of the secondary object.
        :return: The spin of the final object.
        :rtype: array or float
        
        """
        # This is needed to stabilize JAX derivatives
        Seta = np.sqrt(np.where(eta<0.25, 1.0 - 4.0*eta, 0.))
        m1 = 0.5 * (1.0 + Seta)
        m2 = 0.5 * (1.0 - Seta)
        s  = (m1*m1 * chi1 + m2*m2 * chi2)
        af1 = eta*(3.4641016151377544 - 4.399247300629289*eta + 9.397292189321194*eta*eta - 13.180949901606242*eta*eta*eta)
        af2 = eta*(s*((1.0/eta - 0.0850917821418767 - 5.837029316602263*eta) + (0.1014665242971878 - 2.0967746996832157*eta)*s))
        af3 = eta*(s*((-1.3546806617824356 + 4.108962025369336*eta)*s*s + (-0.8676969352555539 + 2.064046835273906*eta)*s*s*s))
        return af1 + af2 + af3
        
    def radiatednrg(eta, chi1, chi2):
        """
        Compute the total radiated energy, as in `arXiv:1508.07250 <https://arxiv.org/abs/1508.07250>`_ eq. (3.7) and (3.8).
        
        :param array or float eta: Symmetric mass ratio of the objects.
        :param array or float chi1: Spin of the primary object.
        :param array or float chi2: Spin of the secondary object.
        :return: Total energy radiated by the system.
        :rtype: array or float
        
        """
        # This is needed to stabilize JAX derivatives
        Seta = np.sqrt(np.where(eta<0.25, 1.0 - 4.0*eta, 0.))
        m1 = 0.5 * (1.0 + Seta)
        m2 = 0.5 * (1.0 - Seta)
        s  = (m1*m1 * chi1 + m2*m2 * chi2) / (m1*m1 + m2*m2)
        
        EradNS = eta * (0.055974469826360077 + 0.5809510763115132 * eta - 0.9606726679372312 * eta*eta + 3.352411249771192 * eta*eta*eta)
        
        return (EradNS * (1. + (-0.0030302335878845507 - 2.0066110851351073 * eta + 7.7050567802399215 * eta*eta) * s)) / (1. + (-0.6714403054720589 - 1.4756929437702908 * eta + 7.304676214885011 * eta*eta) * s)
    
    def RDfreqCalc(finalmass, finalspin, l, m):
        """
        Compute the real and imaginary parts of the complex ringdown frequency for the :math:`(l,m)` mode as in :py:class:`LALSimIMRPhenomHM.c` line 189. This function includes all fits of the different modes.
        
        :param array or float finalmass: Mass(es) of the final object(s).
        :param array or float finalspin: Spin(s) of the final object(s).
        :param int l: :math:`l` of the chosen mode.
        :param int m: :math:`m` of the chosen mode.
        :return: Real and imaginary parts of the complex ringdown frequency (ringdown and damping frequencies).
        :rtype: tuple(array, array) or tuple(float, float)
        
        """
        
        # Domain mapping for dimensionless BH spin
        alpha = np.log(2. - finalspin) / np.log(3.)
        beta = 1. / (2. + l - abs(m))
        kappa  = alpha**beta
        kappa2 = kappa*kappa
        kappa3 = kappa*kappa2
        kappa4 = kappa*kappa3
        
        res = 1.0 + kappa * (1.557847 * np.exp(2.903124 * 1j) + 1.95097051 * np.exp(5.920970 * 1j) * kappa + 2.09971716 * np.exp(2.760585 * 1j) * kappa2 + 1.41094660 * np.exp(5.914340 * 1j) * kappa3 + 0.41063923 * np.exp(2.795235 * 1j) * kappa4)
        
        fring = np.real(res)/(2.*np.pi*finalmass)
        
        fdamp = np.imag(res)/(2.*np.pi*finalmass)
        
        return fring, fdamp
    Mc, eta, chi1, chi2 = evParams['Mc'], evParams['eta'], evParams['chi1z'], evParams['chi2z']
    M = Mc/(eta**(3./5.))
    eta2 = eta*eta
    Seta = np.sqrt(np.where(eta<0.25, 1.0 - 4.0*eta, 0.))
    chi_s = 0.5 * (chi1 + chi2)
    chi_a = 0.5 * (chi1 - chi2)
    # As in arXiv:1508.07253 eq. (4) and LALSimIMRPhenomD_internals.c line 97
    chiPN = (chi_s * (1.0 - eta * 76.0 / 113.0) + Seta * chi_a)
    xi = -1.0 + chiPN
        
    aeff = finalspin(eta, chi1, chi2)
    Erad = radiatednrg(eta, chi1, chi2)
    finMass = 1. - Erad
    
    fring, fdamp = RDfreqCalc(finMass, aeff, 2, 2)
    
    # Compute coefficients gamma appearing in arXiv:1508.07253 eq. (19), the numerical coefficients are in Tab. 5
    gamma1 = 0.006927402739328343 + 0.03020474290328911*eta + (0.006308024337706171 - 0.12074130661131138*eta + 0.26271598905781324*eta2 + (0.0034151773647198794 - 0.10779338611188374*eta + 0.27098966966891747*eta2)*xi+ (0.0007374185938559283 - 0.02749621038376281*eta + 0.0733150789135702*eta2)*xi*xi)*xi
    gamma2 = 1.010344404799477 + 0.0008993122007234548*eta + (0.283949116804459 - 4.049752962958005*eta + 13.207828172665366*eta2 + (0.10396278486805426 - 7.025059158961947*eta + 24.784892370130475*eta2)*xi + (0.03093202475605892 - 2.6924023896851663*eta + 9.609374464684983*eta2)*xi*xi)*xi
    gamma3 = 1.3081615607036106 - 0.005537729694807678*eta +(-0.06782917938621007 - 0.6689834970767117*eta + 3.403147966134083*eta2 + (-0.05296577374411866 - 0.9923793203111362*eta + 4.820681208409587*eta2)*xi + (-0.006134139870393713 - 0.38429253308696365*eta + 1.7561754421985984*eta2)*xi*xi)*xi
    # Compute fpeak, from arXiv:1508.07253 eq. (20), we remove the square root term in case it is complex
    fpeak = np.where(gamma2 >= 1.0, np.fabs(fring - (fdamp*gamma3)/gamma2), fring + (fdamp*(-1.0 + np.sqrt(1.0 - gamma2*gamma2))*gamma3)/gamma2)
    
    return fpeak/(M*glob.GMsun_over_c3)

##############################################################################
# CHECK PARAMETERS
##############################################################################

def check_evparams(evParams):
    """
    Check the format of the events parameters and make the needed conversions.
    
    :param dict(array, array, ...) evParams: Dictionary containing the parameters of the event(s), as in :py:data:`events`.
    
    """
    # Function to check the format of the events' parameters and make the needed conversions
    try:
        _ = evParams['tcoal']
    except KeyError:
        try:
            print('Adding tcoal from tGPS')
            # In the code we use Greenwich Mean Sidereal Time (LMST computed at long = 0. deg) as convention, so convert t_GPS
            evParams['tcoal'] = GPSt_to_LMST(evParams['tGPS'], lat=0., long=0.)
        except KeyError:
            raise ValueError('One among tGPS and tcoal has to be provided.')
    
    try:
        _ = evParams['iota']
    except KeyError:
        try:
            # In the precessing spin case, iota is different from thetaJN, and is computed later. This is just a fix.
            evParams['iota'] = evParams['thetaJN']
        except KeyError:
            raise ValueError('One among iota and thetaJN has to be provided.')
    
    try:
        _ = evParams['Mc']
    except KeyError:
        try:
            print('Adding Mc and eta from the individual detector-frame masses')
            evParams['Mc'], evParams['eta'] = Mceta_from_m1m2(evParams['m1'], evParams['m2'])
        except KeyError:
            raise ValueError('Two among (Mc, eta) and (m1, m2) have to be provided.')
    #try:
    #    _ =evParams['chi1z']
    #except KeyError:
    #    try:
    #        print('Adding chi1z, chi2z from chiS, chiA')
    #        evParams['chi1z'] = evParams['chiS'] + evParams['chiA']
    #        evParams['chi2z'] = evParams['chiS'] - evParams['chiA']
    #    except KeyError:
    #        raise ValueError('Two among chi1z, chi2z and chiS, chiA have to be provided.')
            
    try:
        _ = evParams['theta']
    except KeyError:
        try:
            print('Adding (theta, phi) from (ra, dec)')
            evParams['theta'] = np.pi/2-evParams['dec']
            evParams['phi']=evParams['ra']
        except KeyError:
            raise ValueError('Two among (theta, phi) and (ra, dec) have to be provided.')
    return evParams
                
##############################################################################
# KERR ISCO & POST-MERGER STUFF
##############################################################################

def Kerr_ISCO(Mc, eta, chi1z, chi2z, **kwargs):
    """
    Compute the Kerr ISCO for a rotating final BH, depending on the event parameters, as in `arXiv:2108.05861 <https://arxiv.org/abs/2108.05861>`_ (see in particular App. C).
    
    :param array or float Mc: Chirp mass of the binary, :math:`{\cal M}_c`.
    :param array or float eta: The symmetric mass ratio(s), :math:`\eta`, of the objects.
    :param array or float chi1z: spin of object 1 along the axis :math:`z`, :math:`\chi_{1,z}`.
    :param array or float chi2z: spin of object 2 along the axis :math:`z`, :math:`\chi_{2,z}`.
    :param dict, optional kwargs: Dictionary with arrays containing the other parameters of the events, as in :py:data:`events`.
    
    :return: Kerr ISCO of the final BH for the chosen events, in :math:`\\rm Hz`.
    :rtype: array or float
    """
    eta2 = eta*eta
    Mtot = Mc/(eta**(3./5.))
    chi1, chi2 = chi1z, chi2z
    Seta = np.sqrt(np.where(eta<0.25, 1.0 - 4.0*eta, 0.))
    m1 = 0.5 * (1.0 + Seta)
    m2 = 0.5 * (1.0 - Seta)
    s = (m1*m1 * chi1 + m2*m2 * chi2) / (m1*m1 + m2*m2)
    atot = (chi1 + chi2*(m2/m1)*(m2/m1))/((1.+m2/m1)*(1.+m2/m1))
    aeff = atot + 0.41616*eta*(chi1 + chi2)

    def r_ISCO_of_chi(chi):
        Z1_ISCO = 1.0 + ((1.0 - chi*chi)**(1./3.))*((1.0+chi)**(1./3.) + (1.0-chi)**(1./3.))
        Z2_ISCO = np.sqrt(3.0*chi*chi + Z1_ISCO*Z1_ISCO)
        return np.where(chi>0., 3.0 + Z2_ISCO - np.sqrt((3.0 - Z1_ISCO)*(3.0 + Z1_ISCO + 2.0*Z2_ISCO)), 3.0 + Z2_ISCO + np.sqrt((3.0 - Z1_ISCO)*(3.0 + Z1_ISCO + 2.0*Z2_ISCO)))
    
    r_ISCO = r_ISCO_of_chi(aeff)
    
    EradNS = eta * (0.055974469826360077 + 0.5809510763115132 * eta - 0.9606726679372312 * eta2 + 3.352411249771192 * eta2*eta)
    EradTot = (EradNS * (1. + (-0.0030302335878845507 - 2.0066110851351073 * eta + 7.7050567802399215 * eta2) * s)) / (1. + (-0.6714403054720589 - 1.4756929437702908 * eta + 7.304676214885011 * eta2) * s)
    
    Mfin = Mtot*(1.-EradTot)
    L_ISCO = 2./(3.*np.sqrt(3.))*(1. + 2.*np.sqrt(3.*r_ISCO - 2.))
    E_ISCO = np.sqrt(1. - 2./(3.*r_ISCO))
    
    chif = atot + eta*(L_ISCO - 2.*atot*(E_ISCO - 1.)) + (-3.821158961 - 1.2019*aeff - 1.20764*aeff*aeff)*eta2 + (3.79245 + 1.18385*aeff + 4.90494*aeff*aeff)*eta2*eta

    Om_ISCO = 1./(((r_ISCO_of_chi(chif))**(3./2.))+chif)
    
    return Om_ISCO/(np.pi*Mfin*glob.GMsun_over_c3)

def Mf_of_Mcetachi12(Mc, eta, chi1z, chi2z, **kwargs):
    """
    Compute the final mass :math:`M_f` for a rotating merger remnant BH, depending on the event parameters, as in `arXiv:1508.07250 <https://arxiv.org/abs/1508.07250>`_ (see in particular Sect. 3).
    
    :param array or float Mc: Chirp mass of the binary, :math:`{\cal M}_c`.
    :param array or float eta: The symmetric mass ratio(s), :math:`\eta`, of the objects.
    :param array or float chi1z: spin of object 1 along the axis :math:`z`, :math:`\chi_{1,z}`.
    :param array or float chi2z: spin of object 2 along the axis :math:`z`, :math:`\chi_{2,z}`.
    :param dict, optional kwargs: Dictionary with arrays containing the other parameters of the events, as in :py:data:`events`.
    
    :return: Mass of the final BH for the chosen events, in :math:`{\\rm M}_{\odot}`.
    :rtype: array or float
    """
    eta2 = eta*eta
    Mtot = Mc/(eta**(3./5.))
    chi1, chi2 = chi1z, chi2z
    Seta = np.sqrt(np.where(eta<0.25, 1.0 - 4.0*eta, 0.))
    m1 = 0.5 * (1.0 + Seta)
    m2 = 0.5 * (1.0 - Seta)
    s = (m1*m1 * chi1 + m2*m2 * chi2) / (m1*m1 + m2*m2)
    
    EradNS = eta * (0.055974469826360077 + 0.5809510763115132 * eta - 0.9606726679372312 * eta2 + 3.352411249771192 * eta2*eta)
    EradTot = (EradNS * (1. + (-0.0030302335878845507 - 2.0066110851351073 * eta + 7.7050567802399215 * eta2) * s)) / (1. + (-0.6714403054720589 - 1.4756929437702908 * eta + 7.304676214885011 * eta2) * s)
    
    Mfin = Mtot*(1.-EradTot)
    
    return Mfin

def chif_of_Mcetachi12(Mc, eta, chi1z, chi2z, **kwargs):
    """
    Compute the dimensionless spin parameter of a merger remnant BH :math:`\chi_f`, depending on the event parameters, as in `arXiv:1605.01938 <https://arxiv.org/abs/1605.01938>`_.
    
    :param array or float Mc: Chirp mass of the binary, :math:`{\cal M}_c`.
    :param array or float eta: The symmetric mass ratio(s), :math:`\eta`, of the objects.
    :param array or float chi1z: spin of object 1 along the axis :math:`z`, :math:`\chi_{1,z}`.
    :param array or float chi2z: spin of object 2 along the axis :math:`z`, :math:`\chi_{2,z}`.
    :param dict, optional kwargs: Dictionary with arrays containing the other parameters of the events, as in :py:data:`events`.
    
    :return: Spin of the final BH for the chosen events.
    :rtype: array or float
    """
    eta2 = eta*eta
    chi1, chi2 = chi1z, chi2z
    Seta = np.sqrt(np.where(eta<0.25, 1.0 - 4.0*eta, 0.))
    m1 = 0.5 * (1.0 + Seta)
    m2 = 0.5 * (1.0 - Seta)
    s = (m1*m1 * chi1 + m2*m2 * chi2) / (m1*m1 + m2*m2)
    atot = (chi1 + chi2*(m2/m1)*(m2/m1))/((1.+m2/m1)*(1.+m2/m1))
    aeff = atot + 0.41616*eta*(chi1 + chi2)

    def r_ISCO_of_chi(chi):
        Z1_ISCO = 1.0 + ((1.0 - chi*chi)**(1./3.))*((1.0+chi)**(1./3.) + (1.0-chi)**(1./3.))
        Z2_ISCO = np.sqrt(3.0*chi*chi + Z1_ISCO*Z1_ISCO)
        return np.where(chi>=0., 3.0 + Z2_ISCO - np.sqrt((3.0 - Z1_ISCO)*(3.0 + Z1_ISCO + 2.0*Z2_ISCO)), 3.0 + Z2_ISCO + np.sqrt((3.0 - Z1_ISCO)*(3.0 + Z1_ISCO + 2.0*Z2_ISCO)))
    
    r_ISCO = r_ISCO_of_chi(aeff)

    L_ISCO = 2./(3.*np.sqrt(3.))*(1. + 2.*np.sqrt(3.*r_ISCO - 2.))
    E_ISCO = np.sqrt(1. - 2./(3.*r_ISCO))
    
    chif = atot + eta*(L_ISCO - 2.*atot*(E_ISCO - 1.)) + (-3.821158961 - 1.2019*aeff - 1.20764*aeff*aeff)*eta2 + (3.79245 + 1.18385*aeff + 4.90494*aeff*aeff)*eta2*eta
    
    return chif

class suppress_stdout_stderr(object):
    '''
    A context manager for doing a "deep suppression" of stdout and stderr in
    Python, i.e. will suppress all print, even if the print originates in a
    compiled C/Fortran sub-function.
    This will not suppress raised exceptions, since exceptions are printed
    to stderr just before a script exits, and after the context manager has
    exited.
    
    Full credit goes to https://stackoverflow.com/questions/11130156/suppress-stdout-stderr-print-from-python-functionsorator

    '''
    def __init__(self):
        # Open a pair of null files
        self.null_fds =  [os.open(os.devnull,os.O_RDWR) for x in range(2)]
        # Save the actual stdout (1) and stderr (2) file descriptors.
        self.save_fds = [os.dup(1), os.dup(2)]

    def __enter__(self):
        # Assign the null pointers to stdout and stderr.
        os.dup2(self.null_fds[0],1)
        os.dup2(self.null_fds[1],2)

    def __exit__(self, *_):
        # Re-assign the real stdout/stderr back to (1) and (2)
        os.dup2(self.save_fds[0],1)
        os.dup2(self.save_fds[1],2)
        # Close all file descriptors
        for fd in self.null_fds + self.save_fds:
            os.close(fd)
            
def PatternFunction(lat_rad, long_rad, xax_rad, angbtwArms, ra, dec, t, psi):
    """
    Compute the value of the so-called pattern functions of the detector for a set of sky coordinates, GW polarisation(s) and time(s).
    
    For the definition of the pattern functions see `arXiv:gr-qc/9804014 <https://arxiv.org/abs/gr-qc/9804014>`_ eq. (10)--(13).

    :param array or float ra: The right ascension sky position angle(s), in :math:`\\rm rad`.
    :param array or float dec: The declination sky position angle(s), in :math:`\\rm rad`.
    :param array or float t: The time(s) given as GMST.
    :param array or float psi: The GW polarisation angle(s) :math:`\psi`, in :math:`\\rm rad`.
    :param float rot: Further rotation of the interferometer with respect to the :py:data:`self.xax` orientation, in degrees, needed for the triangular geometry. In this case, the three arms will have orientations 1 --> :py:data:`self.xax`, 2 --> :py:data:`self.xax` + 60°, 3 --> :py:data:`self.xax` + 120°.
    :return: Plus and cross pattern functions of the detector evaluated at the given parameters.
    :rtype: tuple(array, array) or tuple(float, float)
    
    """
    # See P. Jaranowski, A. Krolak, B. F. Schutz, PRD 58, 063001, eq. (10)--(13)
    
    def afun(ra, dec, t):
        phir = long_rad
        a1 = 0.0625*np.sin(2*(xax_rad))*(3.-np.cos(2.*lat_rad))*(3.-np.cos(2.*dec))*np.cos(2.*(ra - phir - 2.*np.pi*t))
        a2 = 0.25*np.cos(2*(xax_rad))*np.sin(lat_rad)*(3.-np.cos(2.*dec))*np.sin(2.*(ra - phir - 2.*np.pi*t))
        a3 = 0.25*np.sin(2*(xax_rad))*np.sin(2.*lat_rad)*np.sin(2.*dec)*np.cos(ra - phir - 2.*np.pi*t)
        a4 = 0.5*np.cos(2*(xax_rad))*np.cos(lat_rad)*np.sin(2.*dec)*np.sin(ra - phir - 2.*np.pi*t)
        a5 = 3.*0.25*np.sin(2*(xax_rad))*(np.cos(lat_rad)*np.cos(dec))**2.
        return a1 - a2 + a3 - a4 + a5
    
    def bfun(ra, dec, t):
        phir = long_rad
        b1 = np.cos(2*(xax_rad))*np.sin(lat_rad)*np.sin(dec)*np.cos(2.*(ra - phir - 2.*np.pi*t))
        b2 = 0.25*np.sin(2*(xax_rad))*(3.-np.cos(2.*lat_rad))*np.sin(dec)*np.sin(2.*(ra - phir - 2.*np.pi*t))
        b3 = np.cos(2*(xax_rad))*np.cos(lat_rad)*np.cos(dec)*np.cos(ra - phir - 2.*np.pi*t)
        b4 = 0.5*np.sin(2*(xax_rad))*np.sin(2.*lat_rad)*np.cos(dec)*np.sin(ra - phir - 2.*np.pi*t)
        
        return b1 + b2 + b3 + b4

    afac = afun(ra, dec, t)
    bfac = bfun(ra, dec, t)

    Fp = np.sin(angbtwArms)*(afac*np.cos(2.*psi) + bfac*np.sin(2*psi))
    Fc = np.sin(angbtwArms)*(bfac*np.cos(2.*psi) - afac*np.sin(2*psi))

    return Fp, Fc
    
def DeltLoc(lat_rad, long_rad, elevation, ra, dec, t):
    """
    Compute the time needed to go from Earth center to detector location for a set of sky coordinates and time(s). The result is given in seconds.

    :param array or float ra: The right ascension sky position angle(s), in :math:`\\rm rad`.
    :param array or float dec: The declination sky position angle(s), in :math:`\\rm rad`.
    :param array or float t: The time(s) given as GMST.
    
    :return: Time shift(s) to go from Earth center to detector location.
    :rtype: array or float
    
    """
    # Time needed to go from Earth center to detector location
    
    semi_major_axis = glob.SMajEarth  # for ellipsoid model of Earth, in km
    semi_minor_axis = glob.SMinEarth  # in km
    radius = semi_major_axis**2 / np.sqrt(semi_major_axis**2 * np.cos(lat_rad)**2 + semi_minor_axis**2 * np.sin(lat_rad)**2)

    comp1 = np.cos(dec)*np.cos(ra)*np.cos(lat_rad)*np.cos(long_rad + 2.*np.pi*t) * (radius + elevation)
    comp2 = np.cos(dec)*np.sin(ra)*np.cos(lat_rad)*np.sin(long_rad + 2.*np.pi*t) * (radius + elevation)
    comp3 = np.sin(dec)*np.sin(lat_rad) * ((semi_minor_axis / semi_major_axis)**2 * radius + elevation)
    # The minus sign arises from the definition of the unit vector pointing to the source
    Delt = - (comp1+comp2+comp3)/glob.clight
    
    return Delt # in seconds