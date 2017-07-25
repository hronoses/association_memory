import matplotlib.pyplot as plt
import numpy as np

N = 100
# graham_original
patt1 =[100, 200, 300, 400, 500]
acc1 = [1.0, 1.0, 0.9933333333333333, 0.9775, 0.786]
# clusteron 2clusters 1syn
patt2 = [100, 200, 300, 400, 500, 600, 700, 800]
acc2 = [1.0, 1.0, 1.0, 0.985, 0.956, 0.925, 0.8657142857142858, 0.78125]

# clusteron 2clusters 5syn

patt3 = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700]
acc3  = [1.0, 1.0, 1.0, 1.0, 0.994, 0.99, 0.9842857142857143, 0.97875, 0.9755555555555555, 0.964, 0.9490909090909091, 0.935, 0.9238461538461539, 0.8978571428571429, 0.868, 0.83625, 0.7947058823529412]

# clusteron 3clusters 5syn

patt4 = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200, 2300, 2400, 2500, 2600, 2700, 2800, 2900]
acc4 = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.9985714285714286, 0.9973333333333333, 0.996875, 0.9952941176470588, 0.9911111111111112, 0.9831578947368421, 0.9765, 0.9638095238095238, 0.9513636363636364, 0.9373913043478261, 0.9216666666666666, 0.9068, 0.8838461538461538, 0.8603703703703703, 0.8389285714285715, 0.8162068965517242]

# clusteron 5clusters 20syn
patt6 = [3000, 3300, 3600, 3900, 4200, 4500, 4800, 5100, 5400, 5700, 6000, 6300, 6600]
acc6 = [1.0, 1.0, 0.9991666666666666, 0.9971794871794872, 0.9902380952380953, 0.9791111111111112, 0.9625, 0.9368627450980392, 0.9066666666666666, 0.8701754385964913, 0.8348333333333333, 0.7988888888888889, 0.7643939393939394]

# corr mat
patt5 = [400, 800, 1200, 1600, 2000, 2400, 2800, 3200, 3600, 4000, 4400, 4800, 5200, 5600, 6000, 6400, 6800, 7200, 7600, 8000, 8400, 8800, 9200, 9600, 10000, 10400, 10800, 11200, 11600, 12000, 12400, 12800, 13200, 13600]
acc5 = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.9998684210526316, 0.99975, 0.9995238095238095, 0.9993181818181818, 0.9992391304347826, 0.9988541666666667, 0.9985, 0.9978846153846154, 0.9959259259259259, 0.9938392857142857, 0.9903448275862069, 0.9845833333333334, 0.9758064516129032, 0.95984375, 0.9445454545454546, 0.9158088235294117]


# plt.plot(patt5, np.array(patt5) * np.array(acc5))
# plt.show()

fig, ax = plt.subplots()
import matplotlib
font_size = 14
matplotlib.rcParams.update({'font.size': font_size})

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
# ax.xaxis.set_ticks_position('bottom')
# ax.yaxis.set_ticks_position('left')
ax.grid(color='k', linestyle=':', linewidth=0.1)

plt.plot(patt1, acc1, 'b', label='Wilshaw')
plt.plot(patt2, acc2, 'r', label='clusteron 2-1 ')
plt.plot(patt3, acc3, 'g', label='clusteron 2-2')
plt.plot(patt4, acc4, 'y', label='clusteron 3-5')
plt.plot(patt6, acc6, 'm', label='clusteron 5-20')
plt.plot(patt5, acc5, 'k', label='triple correlation')
# plt.plot(patt1, acc1, 'k', linestyle='--', label='Wilshaw')
# plt.plot(patt2, acc2, 'k', linestyle='-.',label='clusteron 2-1 ')
# plt.plot(patt3, acc3, 'k', linestyle=':',label='clusteron 2-2')
# plt.plot(patt4, acc4, 'k', linestyle='--',label='clusteron 3-5')
# plt.plot(patt6, acc6, 'k', linestyle='--',label='clusteron 5-20')
# plt.plot(patt5, acc5, 'k', linestyle='--',label='correlation')


plt.ylabel('Accuracy', size=font_size)
plt.xlabel('Number of patterns', size=font_size)
plt.ylim([0.7, 1.1])
plt.legend(loc='lower right')

plt.show()
