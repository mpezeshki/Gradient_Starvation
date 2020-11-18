import numpy as np
import matplotlib.pyplot as plt
plt.figure(figsize=(5, 15))
ax1 = plt.subplot(3, 1, 1)
ax2 = plt.subplot(3, 1, 2)
ax3 = plt.subplot(3, 1, 3)

accs = {}
avgs = {}
for job in [
'logs6_sp_0.0_spw1_5.7',
'logs6_sp_0.03_spw1_5.7',
'logs6_sp_0.1_spw1_0.5',
'logs6_sp_0.1_spw1_10',
'logs6_sp_0.1_spw1_2',
'logs6_sp_0.1_spw1_20',
'logs6_sp_0.1_spw1_5.7',
'logs6_sp_0.1_spw1_5.7_lr4',
'logs6_sp_0.3_spw1_5.7',


# 'logs6_sp_0.0_spw1_5.7',
# 'logs6_sp_0.1_spw2_15_sgd',
# 'logs6_sp_0.1_spw2_20_sgd',
# 'logs6_sp_0.1_spw2_25_sgd',

# 'logs6_sp_0.0_spw1_5.7',
# 'logs6_sp_0.03_spw2_5.7',
# 'logs6_sp_0.1_spw2_0.5',
# 'logs6_sp_0.1_spw2_10',
# 'logs6_sp_0.1_spw2_15',
# 'logs6_sp_0.1_spw2_2',
# 'logs6_sp_0.1_spw2_20',
# 'logs6_sp_0.1_spw2_25',
# 'logs6_sp_0.1_spw2_5.7',
# 'logs6_sp_0.3_spw2_5.7',

# 'logs6_sp_0.0_urw_sgd',
# 'logs6_sp_0.1_urw_sgd',
# 'logs6_sp_0.5_urw_sgd',
# 'logs6_sp_1.0_urw_sgd',
# 'logs6_sp_0.03_urw',
# 'logs6_sp_0.0_urw',
# 'logs6_sp_0.1_urw',
# 'logs6_sp_0.3_urw',

            ]:
    val = False
    file1 = open('/home/pezeshki/scratch/GS/overparam_spur_corr/' + job + '/log.txt', 'r')
    Lines = file1.readlines()
    accs[job] = []
    avgs[job] = []
    for line in Lines:
        if 'Validation:' in line:
            val = True
        if val and 'Blond_Hair = 1, Male = 1  ' in line:
        # if val and 'Blond_Hair = 1, Male = 0' in line:
            accs[job] += [float(line.split(' ')[-1].strip())]
        if val and 'Average acc:' in line:
            avgs[job] += [float(line.split(' ')[-3].strip())]
        if 'Current lr:' in line:
            val = False
    ax1.plot(accs[job], label=job)
    ax2.plot(avgs[job], label=job)
    ax3.plot((np.array(avgs[job]) + np.array(accs[job])) / 2.0, label=job)

plt.legend()
plt.savefig('plot.png', dpi=200)
