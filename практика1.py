import numpy as np
import matplotlib.pyplot as plt


alpha = 0.8
beta = 2 / 3


def rose(t, alpha, beta):
    return alpha * np.cos(beta * t)


t = np.linspace(0, 50 * np.pi, 50000)
z = rose(t, alpha, beta) * np.exp(1j * t)


def f(z):
    return np.exp(2 * z)


w = f(z)


plt.figure(figsize=(14, 7))


plt.subplot(1, 2, 1)
plt.plot(z.real, z.imag, label='Прообраз розы')
plt.xlabel('Re(z)')
plt.ylabel('Im(z)')
plt.title('Прообраз розы')
plt.legend()
plt.axis('equal')
plt.grid(True)
plt.xlim(-1.5, 1.5)
plt.ylim(-1.5, 1.5)


plt.subplot(1, 2, 2)
plt.plot(w.real, w.imag, label='Образ розы', color='r')
plt.xlabel('Re(w)')
plt.ylabel('Im(w)')
plt.title('Образ розы')
plt.legend()
plt.axis('equal')
plt.grid(True)
plt.xlim(-10, 10)
plt.ylim(-10, 10)

plt.tight_layout()
plt.show()