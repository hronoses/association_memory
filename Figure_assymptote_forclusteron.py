import numpy as np
import matplotlib.pyplot as plt

data_clusteron_1 = np.array([100.,  198.,  296., 393.,  480.,  543.,  593.,  606.,  625.,  624.,  628.,  626.,
  624. , 621. , 617. , 617. , 616. , 614.,  611.,  612.,  611.,  611.,  610,  605.,
  604. , 604. , 604. , 604. , 604. , 603., 602.,  602.,  601.,  601.,  601.,  601.,
  601.])

data_clusteron_2 =np.array([100.,   200.,   299.,   398.,   498.,   596.,   692.,   786. ,  867. ,  945.,
                            1018.,  1073.,  1114.,  1124.,  1140.,  1148.,  1126. , 1112. , 1100. , 1081.,
  1069.,  1058.,  1050.,  1040.,  1029.,  1022.,  1013. , 1007. , 1004. ,  998.,   995.,   991.,   987.,   984.,   982.,   980.,   979. ,  975. ,  971. ,  968.,
   968.,   966.,  966. ,  965. ,  964.,   963.,   962.  , 961.  , 961.  , 961.,])
   # 958.,   958.,   958.,   957. ,  957.,   957.,   957. ,  957. ,  956. ,  956.,
   # 955.,   955.,   954.,   953. ,  953.])

data_wilshaw = [50.0, 100.0, 150.0, 200.0, 250.0, 298.0, 348.0, 391.0, 403.0, 393.0, 325.0, 244.0, 147.0, 88.0, 52.0]
data_wilshaw_x = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750]

fig, ax = plt.subplots()

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
# ax.xaxis.set_ticks_position('bottom')
# ax.yaxis.set_ticks_position('left')
ax.grid(color='k', linestyle=':', linewidth=0.1)


plt.plot(np.arange(1, data_clusteron_1.size+1)*100, data_clusteron_1, color='k',  linewidth=1,  label='Clusteron 1 syn')
plt.plot(np.arange(1, data_clusteron_2.size+1)*100, data_clusteron_2, color='k',linestyle=':', linewidth=1, label='Clusteron 2 syn')
plt.plot(data_wilshaw_x, data_wilshaw, color='k', linestyle='-.', linewidth=1, label='Wilshaw model')

plt.axhline(601, color='b', linestyle='--', linewidth=1)
plt.annotate('Max capacity', xy=(0, 601), xytext=(3500, 604), size=15)
plt.xlabel('Number of presented patterns')
plt.ylabel('Number of correctly retrieved patterns')
plt.xticks(np.arange(1, data_clusteron_2.size+1, 5)*100)
plt.legend(loc='lower right', fontsize=15)
# plt.rc("legend", fontsize=20)
plt.show()
# print data