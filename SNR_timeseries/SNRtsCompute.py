import os
import sys

import copy
import numpy as np

import SNRtsGlobals as glob
import SNRtsUtils as utils

class simulate_SNR_timeseries(object):
    def __init__(self, detNet, 
                 is_ASD=False, 
                 fmin=2.,
                 fmax=4096., # in Hz
                 time_interval=20., # in ms
                 df_integrals = 1./4096., # in Hz
                 individual_modes=None,
                 reference_detector=None,
                 ):
        self.detNet = detNet
        self.lat_rad    = {}
        self.long_rad   = {}
        self.xax_rad    = {}
        self.angbtwArms = {}
        self.elevation  = {}
        self.PSDs       = {}
        for det in self.detNet.keys():
            self.lat_rad[det]       = np.deg2rad(self.detNet[det]['lat'])
            self.long_rad[det]      = np.deg2rad(self.detNet[det]['long'])
            self.xax_rad[det]       = np.deg2rad(self.detNet[det]['xax'])
            self.angbtwArms[det]    = 0.5*np.pi if self.detNet[det]['shape'] == 'L' else np.pi/3.
            self.elevation[det]     = self.detNet[det]['elevation']  # in km
            
            noise_read = np.loadtxt(self.detNet[det]['psd_path'], usecols=(0,1))
            if is_ASD:
                noise_read[:,1] = (noise_read[:,1])**2
            self.PSDs[det] = noise_read
        
        self.reference_detector = reference_detector if reference_detector is not None else list(self.detNet.keys())[0]
        
        self.signal_injected = False
        self.fmin = fmin
        self.fmax = fmax
        self.time_interval = time_interval / 1000.  # convert to seconds
        self.df_integrals = df_integrals
        if individual_modes is None:
            self.compute_individual_modes = False
        else:
            self.compute_individual_modes = True
            self.individual_modes = individual_modes
        
    def injectSignal(self, evParams, waveform, df=1./8.):
        
        missing_params = [p for p in waveform.parameters if p not in evParams]
        assert not missing_params, f"Missing parameters: {missing_params}"
        
        tGMST = utils.GPSt_to_LMST(evParams['tGPS'], 0., 0.)
        
        fcut = waveform.fcut(**evParams)
        fcut = min(fcut, self.fmax)
        f = np.arange(self.fmin, fcut + df, df)
        self.fgrid = f
        hp, hc = waveform.hphc(f, **evParams)
        hp, hc = hp*np.exp(1j*(2*np.pi*f*tGMST*3600.*24)), -hc*np.exp(1j*(2*np.pi*f*tGMST*3600.*24))
        
        if self.compute_individual_modes:
            hp_indiv = {}
            tmpparams = copy.deepcopy(evParams)
            tmpparams['iota'] = np.pi*0.5  # Set inclination to 90 degrees for individual modes
            for mode in self.individual_modes:
                l, m = int(mode[0]), int(mode[1])
                tmpparams['modes'] = [[l, m], [l, -m]]
                hp_indiv[mode], _ = waveform.hphc(f, **tmpparams)
                hp_indiv[mode] = hp_indiv[mode]*np.exp(1j*(2*np.pi*f*tGMST*3600.*24))
                
        taus = np.arange(-self.time_interval, self.time_interval + self.df_integrals, self.df_integrals)
        
        # Now compute the SNR time series for each detector
        SNR_ts     = {}
        SNR_sq_opt = {}
        if not self.compute_individual_modes:
            totSNR_sq = 0.
        else:
            totSNR_sq = {}
            for mode in self.individual_modes:
                totSNR_sq[mode] = 0.
            totSNR_sq['all_modes'] = 0.
            modes_correlation = {}
            
        for det in self.detNet.keys():
            strainGrids = np.interp(f, self.PSDs[det][:,0], self.PSDs[det][:,1], left=1., right=1.)
            
            t = tGMST
            tmpDeltLoc = utils.DeltLoc(self.lat_rad[det], self.long_rad[det], self.elevation[det], evParams['ra'], evParams['dec'], t)
            #t = t + tmpDeltLoc/(3600.*24.)
            
            phiL = 2.*np.pi*f*tmpDeltLoc
            # Compute the pattern functions
            Fp, Fc = utils.PatternFunction(self.lat_rad[det], self.long_rad[det], self.xax_rad[det], self.angbtwArms[det], evParams['ra'], evParams['dec'], t, evParams['psi'])
            detsignal = hp * Fp * np.exp(1j*phiL) + hc * Fc * np.exp(1j*phiL)
            
            if not self.compute_individual_modes:
                # first the optimal SNR and autocorrelation of the template
                SNR_sq_opt[det] = abs(4*np.sum(abs(hp)**2 / strainGrids, axis=0) * df)
                SNRs_sq_tot = 4.*np.sum(detsignal[:, None] * np.exp(-2j*np.pi*f[:, None]*taus[None, :]) * np.conjugate(hp)[:, None] / strainGrids[:, None], axis=0) * df
                totSNR_sq += SNR_sq_opt[det]
                # Store the SNR time series
                SNR_ts[det] = SNRs_sq_tot/np.sqrt(SNR_sq_opt[det])

            else:
                SNR_sq_opt[det] = {}
                SNR_ts[det] = {}
                SNR_sq_opt[det]['all_modes'] = 0.
                for mode in self.individual_modes:
                    SNR_sq_opt[det][mode] = abs(4*np.sum(abs(hp_indiv[mode])**2 / strainGrids, axis=0) * df)
                    SNRs_sq_tot = 4.*np.sum(detsignal[:, None] * np.exp(-2j*np.pi*f[:, None]*taus[None, :]) * np.conjugate(hp_indiv[mode])[:, None] / strainGrids[:, None], axis=0) * df
                    SNR_ts[det][mode] = SNRs_sq_tot/np.sqrt(SNR_sq_opt[det][mode])
                    totSNR_sq[mode] += SNR_sq_opt[det][mode]
                    SNR_sq_opt[det]['all_modes'] += SNR_sq_opt[det][mode]
                    totSNR_sq['all_modes'] += SNR_sq_opt[det][mode]
                
                if det == self.reference_detector:
                    for i, mode in enumerate(self.individual_modes):
                        for j, mode2 in enumerate(self.individual_modes):
                            if i < j:
                                modes_correlation[mode + '-' + mode2] = 4*np.sum(hp_indiv[mode] * np.conjugate(hp_indiv[mode2]) / strainGrids, axis=0) * df / np.sqrt(SNR_sq_opt[det][mode] * SNR_sq_opt[det][mode2])


        if not self.compute_individual_modes:
            SNR_sq_opt['net'] = totSNR_sq
        else:
            SNR_sq_opt['net'] = {}
            for mode in self.individual_modes:
                SNR_sq_opt['net'][mode] = totSNR_sq[mode]
            SNR_sq_opt['net']['all_modes'] = totSNR_sq['all_modes']
            SNR_sq_opt['modes_correlation'] = modes_correlation
        
        SNR_ts['tgrid'] = taus + evParams['tGPS']
        
        return SNR_ts, SNR_sq_opt

class modeSNR_ratio(object):
    
    def __init__(self, detNet, 
                 is_ASD=False, 
                 fmin=2.,
                 fmax=4096., # in Hz
                 individual_modes=['22', '33', '44'],
                 reference_detector=None,
                 ):
        self.detNet = detNet
        self.lat_rad    = {}
        self.long_rad   = {}
        self.xax_rad    = {}
        self.angbtwArms = {}
        self.elevation  = {}
        self.PSDs       = {}
        for det in self.detNet.keys():
            self.lat_rad[det]       = np.deg2rad(self.detNet[det]['lat'])
            self.long_rad[det]      = np.deg2rad(self.detNet[det]['long'])
            self.xax_rad[det]       = np.deg2rad(self.detNet[det]['xax'])
            self.angbtwArms[det]    = 0.5*np.pi if self.detNet[det]['shape'] == 'L' else np.pi/3.
            self.elevation[det]     = self.detNet[det]['elevation']  # in km
            
            noise_read = np.loadtxt(self.detNet[det]['psd_path'], usecols=(0,1))
            if is_ASD:
                noise_read[:,1] = (noise_read[:,1])**2
            self.PSDs[det] = noise_read
        
        self.reference_detector = reference_detector if reference_detector is not None else list(self.detNet.keys())[0]
        
        self.signal_injected = False
        self.fmin = fmin
        self.fmax = fmax

        self.individual_modes = individual_modes
            
    def __call__(self, evParams, waveform, df=1./8.):
        
        missing_params = [p for p in waveform.parameters if p not in evParams]
        assert not missing_params, f"Missing parameters: {missing_params}"
        
        fcut = waveform.fcut(**evParams)
        fcut = min(fcut, self.fmax)
        f = np.arange(self.fmin, fcut + df, df)
        self.fgrid = f
    
        
        hp_indiv = {}
        tmpparams = copy.deepcopy(evParams)
        tmpparams['iota'] = np.pi*0.5  # Set inclination to 90 degrees for individual modes
        for mode in self.individual_modes:
            l, m = int(mode[0]), int(mode[1])
            tmpparams['modes'] = [[l, m], [l, -m]]
            hp_indiv[mode], _ = waveform.hphc(f, **tmpparams)
                
        SNR_sq_opt = {}
        
        totSNR_sq = {}
        for mode in self.individual_modes:
            totSNR_sq[mode] = 0.
            
        for det in self.detNet.keys():
            strainGrids = np.interp(f, self.PSDs[det][:,0], self.PSDs[det][:,1], left=1., right=1.)
            SNR_sq_opt[det] = {}
            
            for mode in self.individual_modes:
                SNR_sq_opt[det][mode] = abs(4*np.sum(abs(hp_indiv[mode])**2 / strainGrids, axis=0) * df)
                totSNR_sq[mode] += SNR_sq_opt[det][mode]
            
        SNR_sq_opt['net'] = {}
        for mode in self.individual_modes:
            SNR_sq_opt['net'][mode] = totSNR_sq[mode]
        
        return SNR_sq_opt