import numpy as np
# a = [i for i in np.arange(0, 11, 1)]
# print(a)

x = np.arange(10)

y = np.sin(x)

np.savez(r'.\\steady_extrapolation_draw_equal_heat\\'+'newsave_xy', true=x, pred=y)

npzfile = np.load(r'.\\steady_extrapolation_draw_equal_heat\\'+'newsave_xy.npz')

print(npzfile)
print(npzfile['true'])
print(npzfile['pred'])


