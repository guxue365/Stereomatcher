clc, clear, close all;

boxsize = [9 13 17 21 25];

bpp_cbm005 = [37.80	38.40	43.30	48.60	52.90];
bpp_cbm015 = [38.50	39.70	44.30	49.40	53.50];
bpp_cbm05 = [43.30	49.60	53.20	56.30	59.10];

plot(boxsize, bpp_cbm005, boxsize, bpp_cbm015, boxsize, bpp_cbm05);
legend('CBM 0.05', 'CBM 0.15', 'CBM 0.5');