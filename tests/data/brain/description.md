Test data
=========

8th February 2016

Contact inspection (L0 + couplant).

Sample
------

Aluminium block, 40 mm high
Defect: manufactured notch, 3 x 1 mm

Probe
-----

- Manufacturer: Imasonic
- Serial: 12157 1001
- Type: linear array
- Frequency: 5 MHz
- Pitch: 0.63 mm
- Element size: 0.53 x 15 mm

Acquisition
-----------

- Hardware: Peak NDT Micropulse
- Type: Half Matrix Capture
- Sample frequency: 25 MHz
- Pulse voltage: 100 V
- Pulse width: 80 ns
- Time points: 300
- Sample bits: 8
- Gain (dB): 40
- Filter number: 4
- Maximum PRF (kHz): 2
- Averages: 10 times
- Time start: 5 µs
- Instrument delay: 0 ns

Files
-----
Generated with BRAIN 1.6, Matlab 2014.

Save in Matlab 7.0 format (latest non-HDF5 format) and Matlab 7.3 format (HDF5 format):

    save exp_data -v7
    save exp_data -v7.3

