#
#    Copyright (c) 2022 Francesco Iacovelli <francesco.iacovelli@unige.ch>, Michele Mancarella <michele.mancarella@unige.ch>
#
#    All rights reserved. Use of this source code is governed by the
#    license that can be found in the LICENSE file.

import os
import time

# We use both the original numpy, denoted as onp, and the JAX implementation of numpy, denoted as np
import numpy as np

from abc import ABC, abstractmethod
import os
import sys
import h5py

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(SCRIPT_DIR)

import SNRtsGlobals as glob
import SNRtsUtils as utils

try:
    import warnings
    warnings.filterwarnings("ignore", "Wswiglal-redir-stdio")
    import lal
    import lalsimulation as lalsim
except ModuleNotFoundError:
    print('LSC Algorithm Library (LAL) is not installed, only the GWFAST waveform models are available, namely: TaylorF2, IMRPhenomD, IMRPhenomD_NRTidalv2, IMRPhenomHM and IMRPhenomNSBH')

##############################################################################
# WaveFormModel CLASS DEFINITION
##############################################################################

class WaveFormModel(ABC):
    """
    Abstract class to compute waveforms
    
    :param str objType: The kind of system the wf model is made for, can be ``'BBH'``, ``'BNS'`` or ``'NSBH'``.
    :param float fcutPar: The cut frequency factor of the waveform. This can either be given in :math:`\\rm Hz`, as for :py:class:`gwfast.waveforms.TaylorF2_RestrictedPN`, or as an adimensional frequency (Mf), as for the IMR models.
    :param bool, optional is_newtonian: Boolean specifying if the waveform is a simple Newtonian inspiral.
    :param bool, optional is_tidal: Boolean specifying if the waveform includes tidal effects.
    :param bool, optional is_HigherModes: Boolean specifying if the waveform includes the contribution of sub-dominant (higher-order) modes.
    :param bool, optional is_chi1chi2: Boolean specifying if, in the aligned spins only case, the individual spins are used in place of the ``'chiS'`` and ``'chiA'`` combinations.
    :param bool, optional is_Precessing: Boolean specifying if the waveform includes spin-precession effects.
    :param bool, optional is_LAL: Boolean specifying if the waveform comes from the ``LAL`` library.
    :param bool, optional is_prec_ang: Boolean specifying if, in the precessing spin case, the angular variables of the spins are used, namely ``'thetaJN'``, ``'chi1'``, ``'chi2'``, ``'tilt1'``, ``'tilt2'``, ``'phiJL'``, ``'phi12'``.
    :param bool, optional is_eccentric: Boolean specifying if the waveform includes orbital eccentricity.
    :param bool, optional is_holomorphic: Boolean specifying if the waveform function is holomorphic (needed for derivatives handling).
    :param bool, optional is_cogwheel_params: Boolean specifying if the set of variables from `arXiv:2207.03508 <https://arxiv.org/abs/2207.03508>`_ and `arXiv:2301.04529 <https://arxiv.org/abs/2301.04529>`_ is used.
    :param bool, optional add_aLOS: Boolean specifying if the effect of line-of-sight acceleration at 3.5PN order has to be included in the waveform (it can be added to any model), as in `arXiv:2302.09651 <https://arxiv.org/abs/2302.09651>`_.
    :param bool, optional apply_fcut: Boolean specifying if the waveform has to be cut at the chosen maximum frequency specified by ``fcutPar`` (as in ``LAL``) or not.
    
    """
    
    def __init__(self, objType, fcutPar, is_newtonian=False, is_tidal=False, is_HigherModes=False, is_Precessing=False, is_LAL=False, is_eccentric=False, apply_fcut=True):
        """
        Constructor method
        """
        # The kind of system the wf model is made for, can be 'BBH', 'BNS' or 'NSBH'
        self.objType = objType 
        # The cut frequency factor of the waveform, in Hz, to be divided by Mtot (in units of Msun). The method fcut can be redefined, as e.g. in the IMRPhenomD implementation, and fcutPar can be passed as an adimensional frequency (Mf)
        self.fcutPar = fcutPar
        
        # Dictionary containing the order in which the parameters will appear in the Fisher matrix
        self.ParNums = {'Mc':0, 'eta':1, 'dL':2, 'ra':3, 'dec':4, 'iota':5, 'psi':6, 'tGPS':7, 'Phicoal':8, 'chi1z':9,  'chi2z':10}
        """
        Dictionary containing the number of the rows/columns in which the parameters will appear in the Fisher matrix.
        
        :type: dict(int)
        """
        self.is_newtonian=is_newtonian
        self.is_tidal=is_tidal
        self.is_HigherModes = is_HigherModes
        self.nParams = 11
        self.is_Precessing = is_Precessing
        self.is_LAL = is_LAL
        self.is_eccentric=is_eccentric
        self.apply_fcut = apply_fcut
        
        if is_newtonian:
            # In the Newtonian case eta and the spins are not included in the Fisher, since they do not enter the signal
            self.ParNums = {'Mc':0, 'dL':1, 'ra':2, 'dec':3, 'iota':4, 'psi':5, 'tGPS':6, 'Phicoal':7}
            self.nParams = 8
        if (is_Precessing) and (is_tidal):
            if not is_eccentric:
                self.ParNums = {'Mc':0, 'eta':1, 'dL':2, 'ra':3, 'dec':4, 'iota':5, 'psi':6, 'tGPS':7, 'Phicoal':8, 'chi1z':9,  'chi2z':10, 'chi1x':11, 'chi2x':12, 'chi1y':13, 'chi2y':14, 'LambdaTilde':15, 'deltaLambda':16}
                self.nParams = 17
            else:
                self.ParNums = {'Mc':0, 'eta':1, 'dL':2, 'ra':3, 'dec':4, 'iota':5, 'psi':6, 'tGPS':7, 'Phicoal':8, 'chi1z':9,  'chi2z':10, 'chi1x':11, 'chi2x':12, 'chi1y':13, 'chi2y':14, 'LambdaTilde':15, 'deltaLambda':16, 'ecc':17}
                self.nParams = 18
        elif (is_tidal) and (not is_Precessing):
            # Note that the Fisher is computed for LabdaTilde and deltaLambda, but the waveforms accept as input only Lambda1 and Lambda2
            self.ParNums['LambdaTilde']=11
            self.ParNums['deltaLambda']=12
            if not is_eccentric:
                self.nParams = 13
            else:
                self.ParNums['ecc']=13
                self.nParams = 14
        elif (not is_tidal) and (is_Precessing):
            if not is_eccentric:
                self.ParNums = {'Mc':0, 'eta':1, 'dL':2, 'ra':3, 'dec':4, 'iota':5, 'psi':6, 'tGPS':7, 'Phicoal':8, 'chi1z':9,  'chi2z':10, 'chi1x':11, 'chi2x':12, 'chi1y':13, 'chi2y':14}
                self.nParams = 15
            else:
                self.ParNums = {'Mc':0, 'eta':1, 'dL':2, 'ra':3, 'dec':4, 'iota':5, 'psi':6, 'tGPS':7, 'Phicoal':8, 'chi1z':9,  'chi2z':10, 'chi1x':11, 'chi2x':12, 'chi1y':13, 'chi2y':14, 'ecc':15}
                self.nParams = 16
        elif (not is_tidal) and (not is_Precessing) and (is_eccentric):
            self.ParNums['ecc']=11
            self.nParams = 12
            
        self.ParNums = dict(sorted(self.ParNums.items(), key=lambda item: item[1]))
        self.parameters = list(self.ParNums.keys())
    @abstractmethod    
    def Phi(self, f, **kwargs):
        """
        Compute the phase of the GW as a function of frequency, given the events parameters.

        We compute here only the GW phase, not the full phase of the signal, which also includes the reference phase and the time of coalescence.
        
        :param array f: Frequency grid on which the phase will be computed, in :math:`\\rm Hz`.
        :param dict(array, array, ...) kwargs: Dictionary with arrays containing the parameters of the events to compute the phase of, as in :py:data:`events`.
        :return: GW phase for the chosen events evaluated on the frequency grid.
        :rtype: array
        
        """
        pass
    
    @abstractmethod
    def Ampl(self, f, **kwargs):
        """
        Compute the amplitude of the GW as a function of frequency, given the events parameters.
        
        :param array f: Frequency grid on which the phase will be computed, in :math:`\\rm Hz`.
        :param dict(array, array, ...) kwargs: Dictionary with arrays containing the parameters of the events to compute the amplitude of, as in :py:data:`events`.
        :return: GW amplitude for the chosen events evaluated on the frequency grid.
        :rtype: array
        
        """
        pass
        
    def tau_star(self, f, **kwargs):
        # The relation among the time to coalescence (in seconds) and the frequency (in Hz). We use as default 
        # the expression in M. Maggiore - Gravitational Waves Vol. 1 eq. (4.21), valid in Newtonian and restricted PN approximation
        """
        Compute the time to coalescence (in seconds) as a function of frequency (in :math:`\\rm Hz`), given the events parameters.
        
        :param array f: Frequency grid on which the time to coalescence will be computed, in :math:`\\rm Hz`.
        :param dict(array, array, ...) kwargs: Dictionary with arrays containing the parameters of the events to compute the time to coalescence of, as in :py:data:`events`.
        :return: time to coalescence for the chosen events evaluated on the frequency grid, in seconds.
        :rtype: array
        
        """
        return 2.18567 * ((1.21/kwargs['Mc'])**(5./3.)) * ((100/f)**(8./3.))
    
    def fcut(self, **kwargs):
        """
        Compute the cut frequency of the waveform as a function of the events parameters, in :math:`\\rm Hz`.
        
        :param dict(array, array, ...) kwargs: Dictionary with arrays containing the parameters of the events to compute the cut frequency of, as in :py:data:`events`.
        :return: Cut frequency of the waveform for the chosen events, in :math:`\\rm Hz`.
        :rtype: array
        
        """
        return self.fcutPar/(kwargs['Mc']/(kwargs['eta']**(3./5.)))

##############################################################################
# NEWTONIAN INSPIRAL WAVEFORM
##############################################################################

class NewtInspiral(WaveFormModel):
    """
    Leading order inspiral only waveform model.
    
    Relevant references: `M. Maggiore -- Gravitational Waves Vol. 1 <https://global.oup.com/academic/product/gravitational-waves-9780198570745?q=Michele%20Maggiore&lang=en&cc=it>`_, chapter 4.
        
    
    :param kwargs: Optional arguments to be passed to the parent class :py:class:`WaveFormModel`.
    
    """
    def __init__(self, **kwargs):
        """
        Constructor method
        """
        # Cut from M. Maggiore - Gravitational Waves Vol. 2 eq. (14.106)
        # From T. Dietrich et al. Phys. Rev. D 99, 024029, 2019, below eq. (4) (also look at Fig. 1) it seems be that, for BNS in the non-tidal case, the cut frequency should be lowered to (0.04/(2.*np.pi*glob.GMsun_over_c3))/Mtot.
        super().__init__('BBH', 1./(6.*np.pi*np.sqrt(6.)*glob.GMsun_over_c3), is_newtonian=True, is_holomorphic=True, **kwargs)
    
    def Phi(self, f, **kwargs):
        """
        Compute the phase of the GW as a function of frequency, given the events parameters.

        We compute here only the GW phase, not the full phase of the signal, which also includes the reference phase and the time of coalescence.
        
        :param array f: Frequency grid on which the phase will be computed, in :math:`\\rm Hz`.
        :param dict(array, array, ...) kwargs: Dictionary with arrays containing the parameters of the events to compute the phase of, as in :py:data:`events`.
        :return: GW phase for the chosen events evaluated on the frequency grid.
        :rtype: array
        
        """
        phase = 3.*0.25*(glob.GMsun_over_c3*kwargs['Mc']*8.*np.pi*f)**(-5./3.)
        return phase - np.pi*0.25
    
    def Ampl(self, f, **kwargs):
        """
        Compute the amplitude of the GW as a function of frequency, given the events parameters.
        
        :param array f: Frequency grid on which the phase will be computed, in :math:`\\rm Hz`.
        :param dict(array, array, ...) kwargs: Dictionary with arrays containing the parameters of the events to compute the amplitude of, as in :py:data:`events`.
        :return: GW amplitude for the chosen events evaluated on the frequency grid.
        :rtype: array
        
        """
        amplitude = np.sqrt(5./24.) * (np.pi**(-2./3.)) * glob.clightGpc/kwargs['dL'] * (glob.GMsun_over_c3*kwargs['Mc'])**(5./6.) * (f**(-7./6.))
        return amplitude

##############################################################################
# WRAPPER FOR LAL WAVEFORMS
##############################################################################

class LAL_WF(WaveFormModel):
    """
    Wrapper for using `LAL <https://wiki.ligo.org/Computing/LALSuite>`_ waveforms.
    
    :param str approximant: Name of the waveform model to use, as reported in ``LAL``. If an invalid name is provided, the code will list the names of all the available Fourier domain approximants.
    :param float, optional fcutPar: The cut frequency factor of the waveform as an adimensional frequency (Mf).
    :param bool, optional is_tidal: Boolean specifying if the waveform includes tidal effects.
    :param bool, optional is_HigherModes: Boolean specifying if the waveform includes the contribution of sub-dominant (higher-order) modes.
    :param bool, optional is_Precessing: Boolean specifying if the waveform includes spin-precession effects.
    :param bool, optional is_eccentric: Boolean specifying if the waveform includes orbital eccentricity.
    :param float, optional fRef_ecc: The reference frequency for the provided eccentricity, :math:`f_{e_{0}}`.
    :param float, optional fRef: Reference frequency of the waveform, in :math:`\\rm Hz`. If not provided, the minimum of the frequency grid will be used.
    :param bool, optional compute_sequence: Boolean to specify which ``LAL`` function to use among :py:class:`SimInspiralChooseFDWaveformSequence` (``True``) and :py:class:`SimInspiralChooseFDWaveform` (``False``).
    :param kwargs: Optional arguments to be passed to the parent class :py:class:`WaveFormModel`, such as ``is_chi1chi2``.
    
    """
    '''
    Note that this does not work with JAX computation of derivatives.
    
    NOTE: as a default, we use the LAL function SimInspiralChooseFDWaveformSequence, which computes the waveform on a
    given frequency grid. However, this shows numerical issues with some waveform models (e.g. IMRPhenomXHM), we thus
    also give the possibility to use the function SimInspiralChooseFDWaveform which appears more stable. This performs
    the computation on a LAL defined grid, which then has to be interpolated, resulting in less accurate evaluation at
    low frequencies and slower execution time.
    This can be chosen with the boolean compute_sequence: setting it to True means that the function will perform
    the computation directly on the user grid, setting it to False it will let LAL choose the grid and then extrapolate
    
    '''
    
    def __init__(self, approximant, fcutPar=0.3, is_tidal=False, is_HigherModes=False, is_Precessing=False, is_eccentric=False, compute_sequence=True, fRef_ecc=None, fRef=None, **kwargs):
        """
        Constructor method
        """
        if is_tidal:
            objectT = 'BNS'
        else:
            objectT = 'BBH'
        # approx_name will be filled with the names of all Fourier-domain approximants available in LAL
        approx_name = []
        for approx_enum in range(0, lalsim.NumApproximants):
            if lalsim.SimInspiralImplementedFDApproximants(approx_enum):
                approx_name.append(lalsim.GetStringFromApproximant(approx_enum))
        if approximant not in approx_name:
            raise ValueError('The chosen waveform is not available in LALSimulation, choose one among \n%s'%"\n".join(approx_name))
            
        self.approx = lalsim.GetApproximantFromString(approximant)
        if approximant=='IMRPhenomXHM':
            compute_sequence=False
        self.compute_sequence = compute_sequence
        self.noMbanding = False
        if 'noMbanding' in kwargs.keys():
            self.noMbanding = kwargs['noMbanding']
            kwargs.pop('noMbanding')
        if (is_eccentric) and (self.compute_sequence):
            print('WARNING: SimInspiralChooseFDWaveformSequence function does not accept eccentricity as parameter, resorting to SimInspiralChooseFDWaveform.')
            self.compute_sequence=False
            
        if not self.compute_sequence:
            self.delta_f_base = 1./32.
        self.fRef_ecc = fRef_ecc
        self.fRef = fRef
        super().__init__(objectT, fcutPar, is_tidal=is_tidal, is_HigherModes=is_HigherModes, is_Precessing=is_Precessing, is_eccentric=is_eccentric, is_LAL=True, **kwargs)
    
    def Phi(self, f, **kwargs):
        """
        Compute the phase of the GW as a function of frequency, given the events parameters.
        
        :param array f: Frequency grid on which the phase will be computed, in :math:`\\rm Hz`.
        :param dict(array, array, ...) kwargs: Dictionary with arrays containing the parameters of the events to compute the phase of, as in :py:data:`events`.
        :return: GW phase for the chosen events evaluated on the frequency grid.
        :rtype: array
        
        """
        
        hps, _ = self.hphc(f, **kwargs)
        
        return np.unwrap(np.angle(hps))

    def Ampl(self, f, **kwargs):
        """
        Compute the amplitude of the GW as a function of frequency, given the events parameters.
        
        :param array f: Frequency grid on which the phase will be computed, in :math:`\\rm Hz`.
        :param dict(array, array, ...) kwargs: Dictionary with arrays containing the parameters of the events to compute the amplitude of, as in :py:data:`events`.
        :return: GW amplitude for the chosen events evaluated on the frequency grid.
        :rtype: array
        
        """
        hps, _ = self.hphc(f, **kwargs)
        
        return abs(hps)
    
    def hphc(self, f, **kwargs):
        """
        Compute the plus and cross polarisations of the GW as a function of frequency, given the events parameters.
        
        :param array f: Frequency grid on which the phase will be computed, in :math:`\\rm Hz`.
        :param dict(array, array, ...) kwargs: Dictionary with arrays containing the parameters of the events to compute the phase of, as in :py:data:`events`.
        :return: Plus and cross polarisations of the GW for the chosen events evaluated on the frequency grid.
        :rtype: tuple(array, array)
        
        """
        m1, m2 = utils.m1m2_from_Mceta(kwargs['Mc'], kwargs['eta'])
        phase = kwargs['Phicoal']
        
        if not self.is_Precessing:
            chi1x, chi2x, chi1y, chi2y = m1*0., m1*0., m1*0., m1*0.
        else:
            chi1x, chi2x, chi1y, chi2y = kwargs['chi1x'], kwargs['chi2x'], kwargs['chi1y'], kwargs['chi2y']
        
        if not self.is_tidal:
            lambda1, lambda2 = m1*0., m1*0.
        else:
            lambda1, lambda2 = kwargs['Lambda1'], kwargs['Lambda2']
        
        if not self.is_eccentric:
            ecc = m1*0.
        else:
            ecc = kwargs['ecc']
        
        iota = kwargs['iota']
        
        if 'modes' in kwargs.keys():
            whichmodes = kwargs['modes']
        else:
            whichmodes = None
                
        def LALSimeval(fgrid, m1, m2, chi1x, chi2x, chi1y, chi2y, chi1z, chi2z, dL, iota, lambda1, lambda2, ecc, phase, modes=None):
            # Initialize dictionary for extra parameters (e.g. tidal deformabilities)
            lal_pars = lal.CreateDict()
            
            if self.is_tidal:
                lalsim.SimInspiralWaveformParamsInsertTidalLambda1(lal_pars, lambda1)
                lalsim.SimInspiralWaveformParamsInsertTidalLambda2(lal_pars, lambda2)
                
            if self.is_eccentric:
                if self.fRef_ecc is None:
                    lalsim.SimInspiralWaveformParamsInsertEccentricityFreq(lal_pars, float(np.amin(fgrid)))
                else:
                    lalsim.SimInspiralWaveformParamsInsertEccentricityFreq(lal_pars, float(self.fRef_ecc))
            
            if self.noMbanding:
                lalsim.SimInspiralWaveformParamsInsertPhenomXHMThresholdMband(lal_pars, float(0.))
                lalsim.SimInspiralWaveformParamsInsertPhenomXPHMThresholdMband(lal_pars, float(0.))
            
            if modes is not None:
                modes_array = lalsim.SimInspiralCreateModeArray()
                for l, m in modes:
                    lalsim.SimInspiralModeArrayActivateMode(modes_array, l, m)
                lalsim.SimInspiralWaveformParamsInsertModeArray(lal_pars, modes_array)

            if self.compute_sequence:
                # Here we perform the computation directly on the input grid, which has to be initialized in a LAL readable array
                LAL_frequency_array = lal.CreateREAL8Vector(len(fgrid))
                LAL_frequency_array.data = fgrid
                fRef = 0. if (self.fRef is None) else self.fRef # in Hz
                # Call LAL
                hp, hc = lalsim.SimInspiralChooseFDWaveformSequence(phase, 
                                                                    m1*glob.uMsun, m2*glob.uMsun,
                                                                    chi1x, chi1y, chi1z,
                                                                    chi2x, chi2y, chi2z,
                                                                    fRef, # reference frequency, if set to 0 internally it will be chosen as the minimum frequency of the grid
                                                                    dL*glob.uGpc,
                                                                    iota, # inclination
                                                                    lal_pars,
                                                                    self.approx,
                                                                    LAL_frequency_array)
            
                return np.array(hp.data.data), np.array(hc.data.data)
                
            else:
                fmin, fmax = float(np.amin(fgrid)), float(np.amax(fgrid))
                if fmin==fmax:
                    fmin=fmin/10.
                    fmax=fmax*10.
                # Check that the grid has enough resolution to allow extrapolation
                if (fmax-fmin)/self.delta_f_base > 4.*len(fgrid):
                    delta_f = float(self.delta_f_base)
                else:
                    delta_f = float((fmax-fmin)/(4.*len(fgrid)))
                fRef = fmin if (self.fRef is None) else self.fRef
                hp, hc = lalsim.SimInspiralChooseFDWaveform(m1=m1*glob.uMsun, m2=m2*glob.uMsun,
                                                            S1x = chi1x, S1y = chi1y, S1z = chi1z,
                                                            S2x = chi2x, S2y = chi2y, S2z = chi2z,
                                                            distance = dL*glob.uGpc, inclination = iota,
                                                            phiRef = 0., longAscNodes=0., eccentricity=ecc,
                                                            meanPerAno = 0., deltaF=delta_f, f_min=fmin-delta_f,
                                                            f_max=fmax, f_ref=fRef, LALpars=lal_pars,
                                                            approximant=self.approx)
                
                # In this case the waveform is computed on a grid produced by LAL,
                # starting from 0 and with spacing delta_f. As in PyCBC, given that an interpolation
                # would be problematic due to the rapidly oscillating nature of the function, we output the
                # waveforms at the nearest point for which the evaluation has been performed.
                # This provides a better extrapolation if the LAL grid has sufficient resolution.
                
                # In PyCBC, this is implemented in pycbc.types.frequencyseries -> FrequencySeries.at_at_frequency
                
                # Given that the grid starts from 0 and has spacing delta_f, the closest point to a given
                # frequency fst will be at the index fst/delta_f of the LAL array.
                idxs = np.array((fgrid/delta_f).astype('int'))
                return np.array(hp.data.data)[idxs], np.array(hc.data.data)[idxs]
        
        hps, hcs = np.zeros_like(f).astype('complex64'), np.zeros_like(f).astype('complex64')
        
        if (m1.ndim==0):# | (len(m1)==1):
            if not f.ndim==0:
                hps, hcs = LALSimeval(np.real(f[:]), float(np.real(m1)), float(np.real(m2)), float(np.real(chi1x)), float(np.real(chi2x)), float(np.real(chi1y)), float(np.real(chi2y)), float(np.real(kwargs['chi1z'])), float(np.real(kwargs['chi2z'])), float(np.real(kwargs['dL'])), float(np.real(iota)), float(np.real(lambda1)), float(np.real(lambda2)), float(np.real(ecc)), float(np.real(phase)), modes=whichmodes)
            else:
                hps, hcs = LALSimeval(np.expand_dims(np.real(f), axis=0), float(np.real(m1)), float(np.real(m2)), float(np.real(chi1x)), float(np.real(chi2x)), float(np.real(chi1y)), float(np.real(chi2y)), float(np.real(kwargs['chi1z'])), float(np.real(kwargs['chi2z'])), float(np.real(kwargs['dL'])), float(np.real(iota)), float(np.real(lambda1)), float(np.real(lambda2)), float(np.real(ecc)), float(np.real(phase)), modes=whichmodes)
        else:
            if f.ndim>m1.ndim:
                for i in range(len(m1)):
                    hps[:,i], hcs[:,i] = LALSimeval(np.real(f[:,i]), float(np.real(m1[i])), float(np.real(m2[i])), float(np.real(chi1x[i])), float(np.real(chi2x[i])), float(np.real(chi1y[i])), float(np.real(chi2y[i])), float(np.real(kwargs['chi1z'][i])), float(np.real(kwargs['chi2z'][i])), float(np.real(kwargs['dL'][i])), float(np.real(iota[i])), float(np.real(lambda1[i])), float(np.real(lambda2[i])), float(np.real(ecc[i])), float(np.real(phase[i])), modes=whichmodes)
            else:
                with utils.suppress_stdout_stderr():
                    for i in range(len(m1)):
                        hps[i], hcs[i] = LALSimeval(np.expand_dims(np.real(f[i]), axis=0), float(np.real(m1[i])), float(np.real(m2[i])), float(np.real(chi1x[i])), float(np.real(chi2x[i])), float(np.real(chi1y[i])), float(np.real(chi2y[i])), float(np.real(kwargs['chi1z'][i])), float(np.real(kwargs['chi2z'][i])), float(np.real(kwargs['dL'][i])), float(np.real(iota[i])), float(np.real(lambda1[i])), float(np.real(lambda2[i])), float(np.real(ecc[i])), float(np.real(phase[i])), modes=whichmodes)
        # The following implementation with map and lambda is faster but shows major issues when using multiprocessing
        #LALfun = lambda f, pars : LALSimeval(f, pars[0], pars[1], pars[2], pars[3], pars[4], pars[5], pars[6], pars[7], pars[8], pars[9], pars[10], pars[11], pars[12])
        #resLAL = np.array(list(map(LALfun, np.real(f.T), np.real(np.array([m1, m2, chi1x, chi2x, chi1y, chi2y, kwargs['chi1z'], kwargs['chi2z'], kwargs['dL'], kwargs['iota'], lambda1, lambda2, ecc]).T))))
        #hps, hcs = resLAL[:,0,:].T, resLAL[:,1,:].T
                
        return hps, -hcs
    
    def tau_star(self, f, **kwargs):
        """
        Compute the time to coalescence (in seconds) as a function of frequency (in :math:`\\rm Hz`), given the events parameters.
        
        We use the expression in `arXiv:0907.0700 <https://arxiv.org/abs/0907.0700>`_ eq. (3.8b).
        
        :param array f: Frequency grid on which the time to coalescence will be computed, in :math:`\\rm Hz`.
        :param dict(array, array, ...) kwargs: Dictionary with arrays containing the parameters of the events to compute the time to coalescence of, as in :py:data:`events`.
        :return: time to coalescence for the chosen events evaluated on the frequency grid, in seconds.
        :rtype: array
        
        """
        # We use the expression in arXiv:0907.0700 eq. (3.8b)
        Mtot_sec = kwargs['Mc']*glob.GMsun_over_c3/(kwargs['eta']**(3./5.))
        v = (np.pi*Mtot_sec*f)**(1./3.)
        eta = kwargs['eta']
        eta2 = eta*eta
        
        OverallFac = 5./256 * Mtot_sec/(eta*(v**8.))
        
        t05 = 1. + (743./252. + 11./3.*eta)*(v*v) - 32./5.*np.pi*(v*v*v) + (3058673./508032. + 5429./504.*eta + 617./72.*eta2)*(v**4) - (7729./252. - 13./3.*eta)*np.pi*(v**5)
        t6  = (- 10052469856691./23471078400. + 128./3.*np.pi*np.pi + 6848./105.*np.euler_gamma + (3147553127./3048192. - 451./12.*np.pi*np.pi)*eta - 15211./1728.*eta2 + 25565./1296.*eta2*eta + 3424./105.*np.log(16.*v*v))*(v**6)
        t7  = (- 15419335./127008. - 75703./756.*eta + 14809./378.*eta2)*np.pi*(v**7)
        
        return OverallFac*(t05 + t6 + t7)
    
    def fcut(self, **kwargs):
        """
        Compute the cut frequency of the waveform as a function of the events parameters, in :math:`\\rm Hz`.
        
        :param dict(array, array, ...) kwargs: Dictionary with arrays containing the parameters of the events to compute the cut frequency of, as in :py:data:`events`.
        :return: Cut frequency of the waveform for the chosen events, in :math:`\\rm Hz`.
        :rtype: array
        
        """
        return self.fcutPar/(kwargs['Mc']*glob.GMsun_over_c3/(kwargs['eta']**(3./5.)))

##############################################################################
# TAYLORF2 3.5 RESTRICTED PN WAVEFORM
##############################################################################

class TaylorF2_RestrictedPN(WaveFormModel):
    """
    TaylorF2 restricted PN waveform model, with coefficients up to 3.5 PN. The amplitude is thus the same as in Newtonian approximation, and the model is valid only in the *inspiral*.
    
    This model can include both the contribution of tidal effects at 5 and 6 PN and the contribution of eccentricity up to 3 PN.
    
    Relevant references:
        [1] `arXiv:0907.0700 <https://arxiv.org/abs/0907.0700>`_
        
        [2] `arXiv:1107.1267 <https://arxiv.org/abs/1107.1267>`_
        
        [3] `arXiv:1402.5156 <https://arxiv.org/abs/1402.5156>`_
        
        [4] `arXiv:1601.05588 <https://arxiv.org/abs/1601.05588>`_
        
        [5] `arXiv:1605.00304 <https://arxiv.org/abs/1605.00304>`_
    
    :param float fHigh: The cut frequency factor of the waveform, in :math:`\\rm Hz`. By default this is set to two times the *Innermost Stable Circular Orbit*, ISCO, frequency of a remnant Schwarzschild BH having a mass equal to the total mass of the binary, see :py:data:`gwfast.gwfastGlobals.f_isco`. Another useful value, whose coefficient is provided in :py:data:`gwfast.gwfastGlobals.f_qK`, is the limit of the quasi-Keplerian approximation, defined as in `arXiv:2108.05861 <https://arxiv.org/abs/2108.05861>`_ (see also `arXiv:1605.00304 <https://arxiv.org/abs/1605.00304>`_), which is more conservative than two times the Schwarzschild ISCO.
    :param bool, optional is_tidal: Boolean specifying if the waveform has to include tidal effects.
    :param bool, optional use_3p5PN_SpinHO: Boolean specifying if the waveform has to include the quadratic- and cubic-in-spin contributions at 3.5 PN, which are not included in ``LAL``.
    :param bool, optional phiref_vlso: Boolean specifying if the reference frequency of the waveform has to be set to the *Last Stable Orbit*, LSO, frequency.
    :param bool, optional is_eccentric: Boolean specifying if the waveform has to include orbital eccentricity.
    :param float, optional fRef_ecc: The reference frequency for the provided eccentricity, :math:`f_{e_{0}}`.
    :param str, optional which_ISCO: String specifying if the waveform has to be cut at two times the ISCO frequency of a remnant Schwarzschild (non-rotating) BH or of a Kerr BH, as in `arXiv:2108.05861 <https://arxiv.org/abs/2108.05861>`_ (see in particular App. C), with the fits from `arXiv:1605.01938 <https://arxiv.org/abs/1605.01938>`_. The Schwarzschild ISCO can be selected passing ``'Schw'``, while the Kerr ISCO passing ``'Kerr'``. NOTE: the Kerr option pushes the validity of the model to the limit, and is not the default option.
    :param bool, optional use_QuadMonTid: Boolean specifying if the waveform has to include the spin-induced quadrupole due to tidal effects, with the fits from `arXiv:1608.02582 <https://arxiv.org/abs/1608.02582>`_.
    :param kwargs: Optional arguments to be passed to the parent class :py:class:`WaveFormModel`, such as ``is_chi1chi2``.
    
    """
    
    def __init__(self, fHigh=None, is_tidal=False, use_3p5PN_SpinHO=False, phiref_vlso=False, is_eccentric=False, fRef_ecc=None, which_ISCO='Schw', use_QuadMonTid=False, use_above3p5PN=False, **kwargs):
        """
        Constructor method
        """
        if fHigh is None:
            fHigh = 1./(6.*np.pi*np.sqrt(6.)*glob.GMsun_over_c3) #Hz
        if is_tidal:
            objectT = 'BNS'
        else:
            objectT = 'BBH'
        self.use_3p5PN_SpinHO = use_3p5PN_SpinHO
        self.phiref_vlso = phiref_vlso
        self.fRef_ecc=fRef_ecc
        self.which_ISCO=which_ISCO
        self.use_QuadMonTid = use_QuadMonTid
        self.use_above3p5PN = use_above3p5PN
        super().__init__(objectT, fHigh, is_tidal=is_tidal, is_eccentric=is_eccentric, is_holomorphic=True, **kwargs)
    
    def Phi(self, f, **kwargs):
        """
        Compute the phase of the GW as a function of frequency, given the events parameters.
        
        :param array f: Frequency grid on which the phase will be computed, in :math:`\\rm Hz`.
        :param dict(array, array, ...) kwargs: Dictionary with arrays containing the parameters of the events to compute the phase of, as in :py:data:`events`.
        :return: GW phase for the chosen events evaluated on the frequency grid.
        :rtype: array
        
        """
        # From A. Buonanno, B. Iyer, E. Ochsner, Y. Pan, B.S. Sathyaprakash - arXiv:0907.0700 - eq. (3.18) plus spins as in arXiv:1107.1267 eq. (5.3) up to 2.5PN and PhysRevD.93.084054 eq. (6) for 3PN and 3.5PN
        Mtot_sec = kwargs['Mc']*glob.GMsun_over_c3/(kwargs['eta']**(3./5.))
        v = (np.pi*Mtot_sec*f)**(1./3.)
        eta = kwargs['eta']
        eta2 = eta*eta
        # This is needed to stabilize JAX derivatives
        Seta = np.sqrt(np.where(eta<0.25, 1.0 - 4.0*eta, 0.))
        #Seta = np.sqrt(1.0 - 4.0*eta)
        # These are m1/Mtot and m2/Mtot
        m1ByM = 0.5 * (1.0 + Seta)
        m2ByM = 0.5 * (1.0 - Seta)
        
        chi1, chi2 = kwargs['chi1z'], kwargs['chi2z']
        chi12, chi22 = chi1*chi1, chi2*chi2
        chi1dotchi2  = chi1*chi2
        chi_s, chi_a   = 0.5*(chi1 + chi2), 0.5*(chi1 - chi2)
        chi_s2, chi_a2 = chi_s*chi_s, chi_a*chi_a
        chi_sdotchi_a  = chi_s*chi_a
        # flso = 1/6^(3/2)/(pi*M) -> vlso = (pi*M*flso)^(1/3) = (1/6^(3/2))^(1/3)
        vlso = 1./np.sqrt(6.)
        
        if (self.is_tidal) and (self.use_QuadMonTid):
            Lambda1, Lambda2 = kwargs['Lambda1'], kwargs['Lambda2']
            # A non-zero tidal deformability induces a quadrupole moment (for BBH it is 1).
            # The relation between the two is given in arXiv:1608.02582 eq. (15) with coefficients from third row of Table I
            # We also extend the range to 0 <= Lam < 1, as done in LALSimulation in LALSimUniversalRelations.c line 123
            QuadMon1 = np.where(Lambda1 < 1., 1. + Lambda1*(0.427688866723244 + Lambda1*(-0.324336526985068 + Lambda1*0.1107439432180572)), np.exp(0.1940 + 0.09163 * np.log(Lambda1) + 0.04812 * np.log(Lambda1) * np.log(Lambda1) -4.283e-3 * np.log(Lambda1) * np.log(Lambda1) * np.log(Lambda1) + 1.245e-4 * np.log(Lambda1) * np.log(Lambda1) * np.log(Lambda1) * np.log(Lambda1)))
            QuadMon2 = np.where(Lambda2 < 1., 1. + Lambda2*(0.427688866723244 + Lambda2*(-0.324336526985068 + Lambda2*0.1107439432180572)), np.exp(0.1940 + 0.09163 * np.log(Lambda2) + 0.04812 * np.log(Lambda2) * np.log(Lambda2) -4.283e-3 * np.log(Lambda2) * np.log(Lambda2) * np.log(Lambda2) + 1.245e-4 * np.log(Lambda2) * np.log(Lambda2) * np.log(Lambda2) * np.log(Lambda2)))
        else:
            QuadMon1, QuadMon2 = np.ones(eta.shape), np.ones(eta.shape)
        
        TF2coeffs = {}
        TF2OverallAmpl = 3./(128. * eta)
        
        TF2coeffs['zero'] = 1.
        TF2coeffs['one'] = 0.
        TF2coeffs['two'] = 3715./756. + (55.*eta)/9.
        TF2coeffs['three'] = -16.*np.pi + (113.*Seta*chi_a)/3. + (113./3. - (76.*eta)/3.)*chi_s
        #TF2coeffs['four'] = 15293365./508032. + (27145.*eta)/504.+ (3085.*eta2)/72. + (-405./8. + 200.*eta)*chi_a2 - (405.*Seta*chi_sdotchi_a)/4. + (-405./8. + (5.*eta)/2.)*chi_s2
        # For 2PN coeff we use chi1 and chi2 so to have the quadrupole moment explicitly appearing
        TF2coeffs['four'] = 5.*(3058.673/7.056 + 5429./7.*eta+617.*eta2)/72. + 247./4.8*eta*chi1dotchi2 -721./4.8*eta*chi1dotchi2 + (-720./9.6*QuadMon1 + 1./9.6)*m1ByM*m1ByM*chi12 + (-720./9.6*QuadMon2 + 1./9.6)*m2ByM*m2ByM*chi22 + (240./9.6*QuadMon1 - 7./9.6)*m1ByM*m1ByM*chi12 + (240./9.6*QuadMon2 - 7./9.6)*m2ByM*m2ByM*chi22
        # This part is common to 5 and 5log, avoid recomputing
        TF2_5coeff_tmp = (732985./2268. - 24260.*eta/81. - 340.*eta2/9.)*chi_s + (732985./2268. + 140.*eta/9.)*Seta*chi_a
        if self.phiref_vlso:
            TF2coeffs['five'] = (38645.*np.pi/756. - 65.*np.pi*eta/9. - TF2_5coeff_tmp)*(1.-3.*np.log(vlso))
            phiR = 0.
        else:
            TF2coeffs['five'] = (38645.*np.pi/756. - 65.*np.pi*eta/9. - TF2_5coeff_tmp)
            # This pi factor is needed to include LAL fRef rescaling, so to end up with the exact same waveform
            phiR = np.pi
        TF2coeffs['five_log'] = (38645.*np.pi/756. - 65.*np.pi*eta/9. - TF2_5coeff_tmp)*3.
        #TF2coeffs['six'] = 11583231236531./4694215680. - 640./3.*np.pi**2 - 6848./21.*np.euler_gamma + eta*(-15737765635./3048192. + 2255./12.*np.pi**2) + eta2*76055./1728. - eta2*eta*127825./1296. - (6848./21.)*np.log(4.) + np.pi*(2270.*Seta*chi_a/3. + (2270./3. - 520.*eta)*chi_s) + (75515./144. - 8225.*eta/18.)*Seta*chi_sdotchi_a + (75515./288. - 263245.*eta/252. - 480.*eta2)*chi_a2 + (75515./288. - 232415.*eta/504. + 1255.*eta2/9.)*chi_s2
        # For 3PN coeff we use chi1 and chi2 so to have the quadrupole moment explicitly appearing
        TF2coeffs['six'] = 11583.231236531/4.694215680 - 640./3.*np.pi*np.pi - 684.8/2.1*np.euler_gamma + eta*(-15737.765635/3.048192 + 225.5/1.2*np.pi*np.pi) + eta2*76.055/1.728 - eta2*eta*127.825/1.296 - np.log(4.)*684.8/2.1 + np.pi*chi1*m1ByM*(1490./3. + m1ByM*260.) + np.pi*chi2*m2ByM*(1490./3. + m2ByM*260.) + (326.75/1.12 + 557.5/1.8*eta)*eta*chi1dotchi2 + (4703.5/8.4+2935./6.*m1ByM-120.*m1ByM*m1ByM)*m1ByM*m1ByM*QuadMon1*chi12 + (-4108.25/6.72-108.5/1.2*m1ByM+125.5/3.6*m1ByM*m1ByM)*m1ByM*m1ByM*chi12 + (4703.5/8.4+2935./6.*m2ByM-120.*m2ByM*m2ByM)*m2ByM*m2ByM*QuadMon2*chi22 + (-4108.25/6.72-108.5/1.2*m2ByM+125.5/3.6*m2ByM*m2ByM)*m2ByM*m2ByM*chi22
        TF2coeffs['six_log'] = -(6848./21.)
        if self.use_3p5PN_SpinHO:
        # This part includes SS and SSS contributions at 3.5PN, which are not included in LAL
            TF2coeffs['seven'] = 77096675.*np.pi/254016. + 378515.*np.pi*eta/1512.- 74045.*np.pi*eta2/756. + (-25150083775./3048192. + 10566655595.*eta/762048. - 1042165.*eta2/3024. + 5345.*eta2*eta/36. + (14585./8. - 7270.*eta + 80.*eta2)*chi_a2)*chi_s + (14585./24. - 475.*eta/6. + 100.*eta2/3.)*chi_s2*chi_s + Seta*((-25150083775./3048192. + 26804935.*eta/6048. - 1985.*eta2/48.)*chi_a + (14585./24. - 2380.*eta)*chi_a2*chi_a + (14585./8. - 215.*eta/2.)*chi_a*chi_s2)
        else:
            TF2coeffs['seven'] = 77096675.*np.pi/254016. + 378515.*np.pi*eta/1512.- 74045.*np.pi*eta2/756. + (-25150083775./3048192. + 10566655595.*eta/762048. - 1042165.*eta2/3024. + 5345.*eta2*eta/36.)*chi_s + Seta*((-25150083775./3048192. + 26804935.*eta/6048. - 1985.*eta2/48.)*chi_a)
        if self.use_above3p5PN:
            TF2coeffs['eight'] = -2550713843998885153/2.7680851021824 + 904.9/1.89*np.pi*np.pi + 3681.2/6.3*np.euler_gamma + np.log(2.)*1011.02/1.323 + np.log(3.)*789.75/1.96 + eta*(68071.2846248317/4.224794112 - 1092.95/2.24*np.pi*np.pi + 3911.888/1.323*np.euler_gamma + np.log(2.)*9964.112/1.323 - np.log(3.)*7897.5/4.9) + eta2*(-7510.073635/3.048192 + 112.75/1.44*np.pi*np.pi) - 129.2395/1.2096*eta2*eta+597.5/9.6*eta2*eta2
            TF2coeffs['eight_log'] = 1840.6/6.3 + 1955.944/1.323*eta
            TF2coeffs['nine'] = np.pi*(10534.4279473163/1.877686272 - 640./3.*np.pi*np.pi - 1369.6/2.1*np.euler_gamma - np.log(4.)*1369.6/2.1 + eta*(-14929.17260735/1.34120448 + 2255./6.*np.pi*np.pi) + 452.93335/1.27008*eta2 + 103.23755/1.99584*eta2*eta)
            TF2coeffs['nine_log'] = -1369.6/2.1*np.pi
        else:
            TF2coeffs['eight'] = 0.
            TF2coeffs['eight_log'] = 0.
            TF2coeffs['nine'] = 0.
            TF2coeffs['nine_log'] = 0.

            
        if self.is_eccentric:
            # These are the eccentricity dependent coefficients up to 3 PN order, in the low-eccentricity limit, from arXiv:1605.00304
            ecc = kwargs['ecc']
            if self.fRef_ecc is None:
                v0ecc = np.amin(v, axis=0)
            else:
                v0ecc = (np.pi*Mtot_sec*self.fRef_ecc)**(1./3.)
                
            TF2EccCoeffs = {}
            
            TF2EccOverallAmpl = -2.355/1.462*ecc*ecc*((v0ecc/v)**(19./3.))
            
            TF2EccCoeffs['zero']      = 1.
            TF2EccCoeffs['one']       = 0.
            TF2EccCoeffs['twoV']      = 29.9076223/8.1976608 + 18.766963/2.927736*eta
            TF2EccCoeffs['twoV0']     = 2.833/1.008 - 19.7/3.6*eta
            TF2EccCoeffs['threeV']    = -28.19123/2.82600*np.pi
            TF2EccCoeffs['threeV0']   = 37.7/7.2*np.pi
            TF2EccCoeffs['fourV4']    = 16.237683263/3.330429696 + 241.33060753/9.71375328*eta+156.2608261/6.9383952*eta2
            TF2EccCoeffs['fourV2V02'] = 84.7282939759/8.2632420864-7.18901219/3.68894736*eta-36.97091711/1.05398496*eta2
            TF2EccCoeffs['fourV04']   = -1.193251/3.048192 - 66.317/9.072*eta +18.155/1.296*eta2
            TF2EccCoeffs['fiveV5']    = -28.31492681/1.18395270*np.pi - 115.52066831/2.70617760*np.pi*eta
            TF2EccCoeffs['fiveV3V02'] = -79.86575459/2.84860800*np.pi + 55.5367231/1.0173600*np.pi*eta
            TF2EccCoeffs['fiveV2V03'] = 112.751736071/5.902315776*np.pi + 70.75145051/2.10796992*np.pi*eta
            TF2EccCoeffs['fiveV05']   = 76.4881/9.0720*np.pi - 94.9457/2.2680*np.pi*eta
            TF2EccCoeffs['sixV6']     = -436.03153867072577087/1.32658535116800000 + 53.6803271/1.9782000*np.euler_gamma + 157.22503703/3.25555200*np.pi*np.pi +(2991.72861614477/6.89135247360 - 15.075413/1.446912*np.pi*np.pi)*eta +345.5209264991/4.1019955200*eta2 + 506.12671711/8.78999040*eta2*eta + 384.3505163/5.9346000*np.log(2.) - 112.1397129/1.7584000*np.log(3.)
            TF2EccCoeffs['sixV4V02']  = 46.001356684079/3.357073133568 + 253.471410141755/5.874877983744*eta - 169.3852244423/2.3313007872*eta2 - 307.833827417/2.497822272*eta2*eta
            TF2EccCoeffs['sixV3V03']  = -106.2809371/2.0347200*np.pi*np.pi
            TF2EccCoeffs['sixV2V04']  = -3.56873002170973/2.49880440692736 - 260.399751935005/8.924301453312*eta + 15.0484695827/3.5413894656*eta2 + 340.714213265/3.794345856*eta2*eta
            TF2EccCoeffs['sixV06']    = 265.31900578691/1.68991764480 - 33.17/1.26*np.euler_gamma + 12.2833/1.0368*np.pi*np.pi + (91.55185261/5.48674560 - 3.977/1.152*np.pi*np.pi)*eta - 5.732473/1.306368*eta2 - 30.90307/1.39968*eta2*eta + 87.419/1.890*np.log(2.) - 260.01/5.60*np.log(3.)
            
            phi_Ecc = TF2EccOverallAmpl*(TF2EccCoeffs['zero'] + TF2EccCoeffs['one']*v + (TF2EccCoeffs['twoV']*v*v + TF2EccCoeffs['twoV0']*v0ecc*v0ecc) + (TF2EccCoeffs['threeV']*v*v*v + TF2EccCoeffs['threeV0']*v0ecc*v0ecc*v0ecc) + (TF2EccCoeffs['fourV4']*v*v*v*v + TF2EccCoeffs['fourV2V02']*v*v*v0ecc*v0ecc + TF2EccCoeffs['fourV04']*v0ecc*v0ecc*v0ecc*v0ecc) + (TF2EccCoeffs['fiveV5']*v*v*v*v*v + TF2EccCoeffs['fiveV3V02']*v*v*v*v0ecc*v0ecc + TF2EccCoeffs['fiveV2V03']*v*v*v0ecc*v0ecc*v0ecc + TF2EccCoeffs['fiveV05']*v0ecc*v0ecc*v0ecc*v0ecc*v0ecc) + ((TF2EccCoeffs['sixV6'] + 53.6803271/3.9564000*np.log(16.*v*v))*(v**6) + TF2EccCoeffs['sixV4V02']*v*v*v*v*v0ecc*v0ecc + TF2EccCoeffs['sixV3V03']*v*v*v*v0ecc*v0ecc*v0ecc + TF2EccCoeffs['sixV2V04']*v*v*v0ecc*v0ecc*v0ecc*v0ecc + (TF2EccCoeffs['sixV06'] - 33.17/2.52*np.log(16.*v0ecc*v0ecc))*(v0ecc**6)))
        
        else:
            phi_Ecc = 0.
            
        if self.is_tidal:
            # Add tidal contribution if needed, as in PhysRevD.89.103012
            Lambda1, Lambda2 = kwargs['Lambda1'], kwargs['Lambda2']
            Lam_t, delLam    = utils.Lamt_delLam_from_Lam12(Lambda1, Lambda2, eta)
            
            phi_Tidal = (-0.5*39.*Lam_t)*(v**10.) + (-3115./64.*Lam_t + 6595./364.*Seta*delLam)*(v**12.)
            
        else:
            phi_Tidal = 0.
        
        phase = TF2OverallAmpl*(TF2coeffs['zero'] + TF2coeffs['one']*v + TF2coeffs['two']*v*v + TF2coeffs['three']*v**3 + TF2coeffs['four']*v**4 + (TF2coeffs['five'] + TF2coeffs['five_log']*np.log(v))*v**5 + (TF2coeffs['six'] + TF2coeffs['six_log']*np.log(v))*v**6 + TF2coeffs['seven']*v**7 + (TF2coeffs['eight'] + TF2coeffs['eight_log']*np.log(v))*(v**8)*(np.log(v)) + (TF2coeffs['nine'] + TF2coeffs['nine_log']*np.log(v))*v**9 + phi_Tidal + phi_Ecc)/(v**5.)
            
        return phase + phiR - np.pi*0.25

    def Ampl(self, f, **kwargs):
        """
        Compute the amplitude of the GW as a function of frequency, given the events parameters.
        
        :param array f: Frequency grid on which the phase will be computed, in :math:`\\rm Hz`.
        :param dict(array, array, ...) kwargs: Dictionary with arrays containing the parameters of the events to compute the amplitude of, as in :py:data:`events`.
        :return: GW amplitude for the chosen events evaluated on the frequency grid.
        :rtype: array
        
        """
        # In the restricted PN approach the amplitude is the same as for the Newtonian approximation, so this term is equivalent
        amplitude = np.sqrt(5./24.) * (np.pi**(-2./3.)) * glob.clightGpc/kwargs['dL'] * (glob.GMsun_over_c3*kwargs['Mc'])**(5./6.) * (f**(-7./6.))
        return amplitude
    
    def tau_star(self, f, **kwargs):
        """
        Compute the time to coalescence (in seconds) as a function of frequency (in :math:`\\rm Hz`), given the events parameters.
        
        We use the expression in `arXiv:0907.0700 <https://arxiv.org/abs/0907.0700>`_ eq. (3.8b).
        
        :param array f: Frequency grid on which the time to coalescence will be computed, in :math:`\\rm Hz`.
        :param dict(array, array, ...) kwargs: Dictionary with arrays containing the parameters of the events to compute the time to coalescence of, as in :py:data:`events`.
        :return: time to coalescence for the chosen events evaluated on the frequency grid, in seconds.
        :rtype: array
        
        """
        Mtot_sec = kwargs['Mc']*glob.GMsun_over_c3/(kwargs['eta']**(3./5.))
        v = (np.pi*Mtot_sec*f)**(1./3.)
        eta = kwargs['eta']
        eta2 = eta*eta
        
        OverallFac = 5./256 * Mtot_sec/(eta*(v**8.))
        
        t05 = 1. + (743./252. + 11./3.*eta)*(v*v) - 32./5.*np.pi*(v*v*v) + (3058673./508032. + 5429./504.*eta + 617./72.*eta2)*(v**4) - (7729./252. - 13./3.*eta)*np.pi*(v**5)
        t6  = (- 10052469856691./23471078400. + 128./3.*np.pi*np.pi + 6848./105.*np.euler_gamma + (3147553127./3048192. - 451./12.*np.pi*np.pi)*eta - 15211./1728.*eta2 + 25565./1296.*eta2*eta + 3424./105.*np.log(16.*v*v))*(v**6)
        t7  = (- 15419335./127008. - 75703./756.*eta + 14809./378.*eta2)*np.pi*(v**7)
        
        return OverallFac*(t05 + t6 + t7)
    
    def fcut(self, **kwargs):
        """
        Compute the cut frequency of the waveform as a function of the events parameters, in :math:`\\rm Hz`.
        
        This can be approximated as 2 f_ISCO for inspiral only waveforms. The flag which_ISCO controls the expression of the ISCO to use:
        
            - if ``'Schw'`` is passed the Schwarzschild ISCO for a non-rotating final BH is used (depending only on ``'Mc'`` and ``'eta'``);
            - if ``'Kerr'`` is passed the Kerr ISCO for a rotating final BH is computed (depending on ``'Mc'``, ``'eta'`` and the spins), as in `arXiv:2108.05861 <https://arxiv.org/abs/2108.05861>`_ (see in particular App. C). NOTE: this is pushing the validity of the model to the limit, and is not the default option.
        
        :param dict(array, array, ...) kwargs: Dictionary with arrays containing the parameters of the events to compute the cut frequency of, as in :py:data:`events`.
        :return: Cut frequency of the waveform for the chosen events, in :math:`\\rm Hz`.
        :rtype: array
        
        """
        
        if self.which_ISCO=='Schw':
            
            return self.fcutPar/(kwargs['Mc']/(kwargs['eta']**(3./5.)))
        
        elif self.which_ISCO=='Kerr':
            
            eta = kwargs['eta']
            eta2 = eta*eta
            Mtot = kwargs['Mc']/(eta**(3./5.))
            chi1, chi2 = kwargs['chi1z'], kwargs['chi2z']
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