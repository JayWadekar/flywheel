# Example Data

This directory contains a compact one-event example for
`examples/low_latency_pe.py`.

The files are generated analysis products:

- `snr_timeseries/`: mode-by-mode SNR time series, optimal-SNR metadata, and
  injection/event parameters.
- `mode_ratios/`: intrinsic-parameter samples and sampled 33/22 and 44/22
  amplitude-ratio inputs for the coherent-score marginalization.
- `psds/`: compact LIGO/Virgo design sensitivity curves used by the SNR
  generation smoke test.

The full population data products used in the paper are intentionally not
included here.
