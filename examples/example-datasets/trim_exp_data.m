% Script to trim an exp_data file.
% Make small example datasets on git repo!
clear all
raw = load('D:\Nicolas\Documents\Python\arim\examples\example-datasets\immersion_notch_aluminium_raw.mat');
raw = load('D:\Nicolas\Documents\Python\arim\examples\example-datasets\contact_notch_aluminium_raw.mat');
%%
exp_data = raw.exp_data;
maxelt = 64;
valid_idx = (exp_data.tx <= maxelt) & (exp_data.rx <= maxelt);
exp_data.tx = exp_data.tx(valid_idx);
exp_data.rx = exp_data.rx(valid_idx);
exp_data.time_data = exp_data.time_data(:,valid_idx);
exp_data.array.el_x1 = exp_data.array.el_x1(1:maxelt);
exp_data.array.el_x2 = exp_data.array.el_x2(1:maxelt);
exp_data.array.el_y1 = exp_data.array.el_y1(1:maxelt);
exp_data.array.el_y2 = exp_data.array.el_y2(1:maxelt);
exp_data.array.el_z1 = exp_data.array.el_z1(1:maxelt);
exp_data.array.el_z2 = exp_data.array.el_z2(1:maxelt);
exp_data.array.el_xc = exp_data.array.el_xc(1:maxelt);
exp_data.array.el_yc = exp_data.array.el_yc(1:maxelt);
exp_data.array.el_zc = exp_data.array.el_zc(1:maxelt);

dx = mean(exp_data.array.el_xc);
exp_data.array.el_xc = exp_data.array.el_xc - dx;
exp_data.array.el_x1 = exp_data.array.el_x1 - dx;
exp_data.array.el_x2 = exp_data.array.el_x2 - dx;


%save immersion_notch_aluminium.mat exp_data
save contact_notch_aluminium.mat exp_data
