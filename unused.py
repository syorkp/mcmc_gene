import matplotlib.pyplot as plt
import numpy as np



def plot_autocorrelation(s_tau, s_theta, s):
    # Plot the autocorrelation
    autocorrelation_tau = np.correlate(s_tau, s_tau, mode="full")
    autocorrelation_tau = autocorrelation_tau[autocorrelation_tau.size // 2:]
    plt.plot(autocorrelation_tau, color="b", label="tau")
    autocorrelation_theta = np.correlate(s_theta, s_theta, mode="full")
    autocorrelation_theta = autocorrelation_theta[autocorrelation_theta.size // 2:]
    plt.plot(autocorrelation_theta, color="r", label="theta")
    plt.title(f"Autocorrelation s={s}")
    plt.legend(loc="upper right")
    plt.show()
    # TODO: Calculate burn in from autocorrelation

