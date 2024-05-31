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

theta_z = np.angle(z)
rho_z = np.abs(z)

theta_w = np.angle(w)
rho_w = np.abs(w)

fig, ax = plt.subplots(1, 2, subplot_kw={'projection': 'polar'}, figsize=(14, 7))

ax[0].plot(theta_z, rho_z, label='Прообраз розы')
ax[0].set_title('Прообраз розы в полярной системе')
ax[0].legend()
ax[0].grid(True)

ax[1].plot(theta_w, rho_w, label='Образ розы', color='r')
ax[1].set_title('Образ розы в полярной системе')
ax[1].legend()
ax[1].grid(True)

plt.tight_layout()
plt.show()