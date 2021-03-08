import numpy as np
import matplotlib.pyplot as plt
import os

# Lcurve

dir='../ismipc_rc_1e6/output/'

gam = np.load(dir + 'LCurvGam.npy')
norm = np.load(dir + 'LCurvNorm.npy')
mis = np.load(dir + 'LCurvMis.npy')

fig,ax = plt.subplots()

for i in range(len(gam)):
  ax.plot(norm[i],mis[i],'o',markersize=4,markerfacecolor='blue',markeredgecolor='k')

for i in range(len(gam)):
  if (i==4):
   ax.annotate(str(gam[i]),(norm[i]-6,mis[i]-6))
  else:
   ax.annotate(str(gam[i]),(norm[i]+1,mis[i]+1))

plt.xlabel('Regularisation cost')
plt.ylabel('Model-data misfit cost')


plt.savefig('L_curve.png')

plt.close(fig)

pwd = os.getcwd()

os.system('python ../../aux/plotting/plot_inv_results_hist.py ismipc_rc_1e6 ' + pwd)
os.system('python ../../aux/plotting/plot_inv_results_hist.py ismipc_rc_1e4 ' + pwd)

plt.close(fig)

os.system('python ../../aux/plotting/plot_leading_eigenfuncs.py ' + pwd)

plt.close(fig)

os.system("python ../../aux/plotting/plotting_new/plot_eigenvalue_decay_gen_new.py $PWD reg_decay.png ismipc_rc_1e6 '$\gamma_{alpha}$=50' ismipc_rc_1e5 '$\gamma_{alpha}$=10' ismipc_rc_1e4 '$\gamma_{alpha}$=1'")

plt.close(fig)

os.system("python ../../aux/plotting/plotting_new/plot_eigenvalue_decay_gen_new.py $PWD res_decay.png ismipc_20x20 '$\Delta$ x = 2 km' ismipc_rc_1e5 '$\Delta$ x = 1.33 km' ismipc_40x40 '$\Delta$ x = 1 km'")

plt.close(fig)

os.system("python ../../aux/plotting/plotting_new/plot_eigenvalue_decay_gen_new.py $PWD sample_decay.png ismipc_8000 8000m ismipc_4000 4km ismipc_2000 2km ismipc_1000 1km ismipc_500 500m")

plt.close(fig)

#os.system("python ../../aux/plotting/plotting_new/plot_eigenvalue_decay_gen_new.py $PWD sample_decay_autocorr_alt2.png corr_alt/ismipc_8000_corr 8km corr_alt/ismipc_4000_corr 4km corr_alt/ismipc_2000_corr 2km corr_alt/ismipc_1000_corr 1km corr_alt/ismipc_500_corr 500m")

#plt.close(fig)

os.system('python ../../aux/plotting/plotting_new/plot_paths_sample.py $PWD paths.png')
