frequencies = [1e6, 2e6];
numangles = 11;
numfreq = length(frequencies);
scattering_LL = zeros(numfreq, numangles, numangles);
scattering_LL(1, 1, :) = 1j;
scattering_LL(2, 1, :) = 2j;

scattering_TL = zeros(numfreq, numangles, numangles);
scattering_LL(1, 1, :) = 10j;
scattering_TL(2, 1, :) = 20j;

save -v7 scat_matlab.mat frequencies scattering_LL scattering_TL
