import matplotlib.pyplot as plt

x = [2, 500, 1000, 1500, 2000, 2500]
y1 = [0.946493, 1.233977, 1.768053, 2.118188, 2.633266, 3.416748]
y2 = [0.687389, 0.624856, 0.758891, 0.827954, 0.867985, 1.109116]
y3 = [0.390536, 7.148036, 14.149353, 21.929082, 27.070931, 34.485844]
y4 = [2.252813, 14.489502, 26.731936, 39.361620, 50.967915, 65.345407]
y5 = [2.689894, 8.658210, 14.034492, 20.187914, 25.134806, 31.889481]
y6 = [0.181074, 6.523015, 12.930813, 17.344285, 25.564698, 29.411268]
y7 = [0.171837, 0.779592, 1.461803, 2.077634, 2.868516, 3.523895]
y8 = [0.484260, 1.744686, 3.209689, 4.666350, 6.241945, 7.988499]
y9 = [1.618020, 2.142908, 3.064400, 3.781884, 4.651532, 5.434087]
y10 = [0.650405, 1.228955, 1.901776, 2.695548, 3.667338, 4.201932]

plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False

plt.figure(figsize=(10, 7))
plt.xlabel("维度", fontsize=30)
plt.ylabel('运行时间 (s)', fontsize=30)
# plt.xlim()
# plt.ylim()
plt.xticks(x, fontsize=30)
plt.yticks(fontsize=30)
# plt.yticks([3, 4, 5, 6, 7, 8, 9, 10], fontsize=22)
# plt.title('', fontsize=22)
plt.plot(x, y1, marker='*', color='blue', linewidth=3.0, ms=12, label=r'$\rho_{Anchor}$ [k-means]')
plt.plot(x, y2, marker='^', color='purple', linewidth=3.0, ms=12, label=r'$\rho_{Anchor}$ [random]')
plt.plot(x, y3, marker='X', color='orange', linewidth=3.0, ms=12, label=r'$\rho_{Naive}$')
plt.plot(x, y4, marker='s', color='red', linewidth=3.0, ms=12, label=r'$\rho_{LC}$')
plt.plot(x, y5, marker='<', color='black', linewidth=3.0, ms=12, label=r'$\rho_{FKD}$')
plt.plot(x, y6, marker='>', color='green', linewidth=3.0, ms=12, label=r'$\rho_{RKNN}$')
plt.legend(loc=2, fontsize=23)
plt.tight_layout()
plt.show()
plt.savefig('rho_time1_x'+'.pdf')
plt.clf()
plt.cla()
plt.close()

plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False

plt.figure(figsize=(10, 7))
plt.xlabel("维度", fontsize=30)
plt.ylabel('运行时间 (s)', fontsize=30)
# plt.xlim()
# plt.ylim()
plt.xticks(x, fontsize=30)
plt.yticks(fontsize=30)
# plt.yticks([3, 4, 5, 6, 7, 8, 9, 10], fontsize=22)
# plt.title('', fontsize=22)
plt.plot(x, y1, marker='*', color='blue', linewidth=3.0, ms=12, label=r'$\rho_{Anchor}$ [k-means]')
plt.plot(x, y2, marker='^', color='purple', linewidth=3.0, ms=12, label=r'$\rho_{Anchor}$ [random]')
plt.plot(x, y7, marker='X', color='orange', linewidth=3.0, ms=12, label=r'$\rho_{Naive}$')
plt.plot(x, y8, marker='s', color='red', linewidth=3.0, ms=12, label=r'$\rho_{LC}$')
plt.plot(x, y9, marker='<', color='black', linewidth=3.0, ms=12, label=r'$\rho_{FKD}$')
plt.plot(x, y10, marker='>', color='green', linewidth=3.0, ms=12, label=r'$\rho_{RKNN}$')
plt.legend(loc=2, fontsize=23)
plt.tight_layout()
plt.show()
plt.savefig('rho_time2_x'+'.pdf')