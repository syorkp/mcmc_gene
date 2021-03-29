import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
import random
import cProfile


def run_mcmc(w_tau, w_theta, w_t, tau, theta, t, data, nsample=10000):

    # Uncomment to identify computational bottlenecks:
    # profile = cProfile.Profile()
    # profile.enable()

    # Create values to return
    sample_tau = []
    sample_theta = []
    accept_tau = 0
    accept_theta = 0
    accept_ti = 0
    running_accept_tau = []
    running_accept_theta = []
    running_accept_t = []

    x = data["xi"].tolist()
    n = data["ni"].tolist()

    # Add the init values to the store
    sample_tau.append(tau)
    sample_theta.append(theta)

    # Values stored to reduce calculations - updated independently.
    pre_sum_term = calculate_presum_term(tau, theta)
    lg_2theta = np.log(2/theta)
    sum_terms = calculate_sum_terms(tau, theta, t, x, n, lg_2theta)  # A list of 1-L values for sum term of posterior.
    summed_terms = sum(sum_terms)

    # Calculate the log unnormalised posterior for the init values.
    lup = pre_sum_term + summed_terms

    # Loop over the sample
    for i in range(nsample):
        # -----------           TAU           -----------
        # Change tau and ensure it is a positive value.
        new_tau = tau + (random.random() - 0.5) * w_tau
        if new_tau < 0:
            new_tau = -1 * new_tau

        # Compute new components for the posterior that depend on tau.
        new_pre_sum_term = calculate_presum_term(new_tau, theta)
        new_sum_terms = calculate_sum_terms(new_tau, theta, t, x, n, lg_2theta)

        # Calculate the new log unnormalised posterior.
        summed_terms = sum(new_sum_terms)
        new_lup = new_pre_sum_term + summed_terms

        ratio = new_lup - lup
        # Accept the new tau value with the probability of the ratio of new to old posterior.
        if ratio >= 0 or random.random() < np.exp(ratio):
            tau = new_tau
            accept_tau += 1
            lup = new_lup
            sum_terms = new_sum_terms

        # -----------           THETA           -----------
        # Change theta and ensure it is a positive value.
        new_theta = theta + (random.random() - 0.5) * w_theta
        if new_theta < 0:
            new_theta = -1 * new_theta

        # Compute new components for the posterior that depend on theta.
        new_lg_2theta = np.log(2 / new_theta)
        new_pre_sum_term = calculate_presum_term(tau, new_theta)
        new_sum_terms = calculate_sum_terms(tau, new_theta, t, x, n, new_lg_2theta)

        # Calculate the new log unnormalised posterior.
        summed_terms = sum(new_sum_terms)
        new_lup = new_pre_sum_term + summed_terms

        ratio = new_lup - lup
        # Accept the new theta value with the probability of the ratio of new to old posterior.
        if ratio >= 0 or random.random() < np.exp(ratio):
            theta = new_theta
            accept_theta += 1
            sum_terms = new_sum_terms
            lg_2theta = new_lg_2theta

        # -----------           ti           -----------

        # Loop through and change each coalescent time.
        j = 0
        for xi, ni, ti in zip(x, n, t):
            # Change ti and ensure it is a positive value.
            t_new = ti + (random.random() - 0.5) * w_t
            if t_new < 0:
                t_new = -1 * t_new

            # Compute new components for the posterior that depend on ti.
            old_sum_term = sum_terms[j]
            new_sum_term = update_sum_term(theta, tau, t_new, xi, ni, lg_2theta)
            # summed_terms = summed_terms + new_sum_term - old_sum_term

            # Calculate the new log unnormalised posterior.
            # new_lup = pre_sum_term + summed_terms

            # ratio = new_lup - lup

            ratio = new_sum_term - old_sum_term
            # Accept the new theta value with the probability of the ratio of new to old posterior.
            if ratio >= 0 or random.random() < np.exp(ratio):
                t[j] = t_new
                accept_ti += 1
                sum_terms[j] = new_sum_term
                lup = lup + new_sum_term - old_sum_term
            j += 1

        # Add the values to the samples
        sample_tau.append(tau)
        sample_theta.append(theta)
        if i % 100 == 0:
            # Uncomment to keep track of progress:
            print(i)
            running_accept_tau.append(accept_tau/1)
            running_accept_theta.append(accept_theta/1)
            running_accept_t.append(accept_theta/(1*1000))
            accept_tau = 0
            accept_theta = 0
            accept_ti = 0

    # Uncomment if identifying computational bottlenecks:
    # profile.disable()
    # ps = pstats.Stats(profile)
    # ps.print_stats()

    return sample_tau, sample_theta, running_accept_tau, running_accept_theta, running_accept_t


def calculate_presum_term(tau, theta, mu_tau=0.005, mu_theta=0.001):
    """
    Returns the value for the term -(1/mu_tau)*tau -(1/mu_theta)*theta
    """
    return -(tau/mu_tau) -(theta/mu_theta)


def update_sum_term(theta, tau, ti, xi, ni, lg_2theta):
    """
    Returns the sum term for ti, according to the equation:
    log(2/theta) - 2ti/theta + xilog(3/4 - (3/4)e^(-(8/3)(tau+ti)))) + (ni-xi)log(1/4 + (3/4)e^(-(8/3)(tau+ti))))
    """
    return lg_2theta - (2/theta)*ti + (xi * np.log(3/4-3*(np.exp(-8*(tau+ti)/3))/4)) + ((ni-xi) * np.log(1/4+3*(np.exp(-8*(tau+ti)/3))/4))


def calculate_sum_terms(tau, theta, t, x, n, lg_2theta):
    sum_terms = []
    for xi, ni, ti in zip(x, n, t):
        sum_terms.append(lg_2theta - (2/theta)*ti + (xi * np.log(3/4-3*(np.exp(-8*(tau+ti)/3))/4)) +
                         ((ni-xi) * np.log(1/4+3*(np.exp(-8*(tau+ti)/3))/4)))
    return sum_terms


def load_data(cutoff=1000):
    df = pd.read_csv('HC.SitesDiffs.txt', delimiter="	")
    df = df.iloc[[i for i in range(cutoff)], :]
    df["p"] = [df["xi"][i]/df["ni"][i] for i, row in df.iterrows()]
    df["ti"] = [df["p"][i]/2 for i, row in df.iterrows()]
    return df


def plot_acceptance_ratios(p_accept_tau, p_accept_theta, p_accept_t, s):
    plt.plot(p_accept_tau[1:], color="r")
    plt.plot(p_accept_theta[1:], color="b")
    plt.title(f"Tau and theta acceptance ratios for s={s}")
    plt.show()

    plt.plot(p_accept_t[1:], color="g")
    plt.title(f"t acceptance ratios for s={s}")
    plt.show()


def plot_solution_distibution(s_tau, s_theta, s, burn_in_tau=100, burn_in_theta=100):
    plt.step([i for i in range(len(s_tau))], s_tau)
    plt.title(f"Convergence trace for tau for s={s}")
    plt.axhline(y=np.mean(s_tau[burn_in_tau:]), color="r", linestyle='-', label="Mean")
    plt.axvline(x=burn_in_tau, label='Burn in', color="g")
    plt.legend()
    plt.show()

    plt.step([i for i in range(len(s_theta))], s_theta)
    plt.title(f"Convergence trace for theta for s={s}")
    plt.axhline(y=np.mean(s_theta[burn_in_theta:]), color="r", linestyle='-', label="Mean")
    plt.axvline(x=burn_in_theta, label='Burn in', color="g")
    plt.legend()
    plt.show()

    # Create histograms
    ci_tau = stats.norm(*stats.norm.fit(s_tau)).interval(0.95)
    sns.distplot(s_tau)
    plt.axvline(ci_tau[0], color="g", label="95% CI")
    plt.axvline(ci_tau[1], color="g")
    plt.axvline(np.mean(s_tau[burn_in_tau:]), color="r", label="Mean")
    plt.legend(loc="upper right")
    plt.title(f"Density histogram for tau for s={s}")
    plt.show()

    ci_theta = stats.norm(*stats.norm.fit(s_theta)).interval(0.95)
    sns.distplot(s_theta)
    plt.axvline(ci_theta[0], color="g", label="95% CI")
    plt.axvline(ci_theta[1], color="g")
    plt.axvline(np.mean(s_theta[burn_in_theta:]), color="r", label="Mean")
    plt.legend(loc="upper right")
    plt.title(f"Density histogram for theta for s={s}")
    plt.show()

    print(f"Burn in tau: {burn_in_tau}")
    print(f"Burn in theta: {burn_in_theta}")
    print(f"Mean tau: {np.mean(s_tau[burn_in_tau:])}")
    print(f"Mean theta: {np.mean(s_theta[burn_in_theta:])}")
    print(f"Confidence intervals tau: {ci_tau}")
    print(f"Confidence intervals theta: {ci_theta}")


def get_efficiency(s_tau, s_theta):
    # Calculate measure of efficiency.
    acs_tau = [np.correlate(s_tau[:-i], s_tau[i:]) for i in range(1, len(s_tau))]
    E_tau = 1/(1 + 2*(sum(acs_tau)))
    acs_theta = [np.correlate(s_theta[:-i], s_theta[i:]) for i in range(1, len(s_theta))]
    E_theta = 1/(1 + 2*(sum(acs_theta)))
    return E_tau, E_theta


def get_burn_in_duration(estimates):
    differences = [a-estimates[i-1] for i, a in enumerate(estimates)]
    run_average_diff_sign = []
    for i in range(100, len(estimates)):
        positive = len([a for a in differences[i-100: i] if a > 0])
        negative = len([a for a in differences[i-100: i] if a < 0])
        run_average_diff_sign.append(positive-negative)
    for i, a in enumerate(run_average_diff_sign):
        if a == 0:
            return i + 100
    return None


def create_joint_distribution(tau, theta):
    plt.hist2d(tau, theta, bins=100)
    plt.show()


def scale_search(scaling_factors, nsample=10000):
    # Create results data frame
    search_results = pd.DataFrame(columns=["w_tau", "w_theta", "w_t", "E Tau", "E Theta", "Tau Acceptance",
                                  "Theta Acceptance", "t Acceptance", "Tau Estimate", "Theta Estimate",
                                           "Tau CI (Lower)", "Tau CI (Upper)", "Theta CI (Lower)", "Theta CI (Upper)"])
    sitesdiff = load_data()
    t_init = [0.001 for i in range(1000)]

    # Loop over scaling factors to try.
    for i, s in enumerate(scaling_factors):
        if s > 0.1:
            num_sample = 1000
        else:
            num_sample = nsample
        s_tau, s_theta, p_accept_tau, p_accept_theta, p_accept_t = run_mcmc(w_tau=0.01*s, w_theta=0.01*s, w_t=0.002*s,
                                                                            tau=0.01, theta=0.001,
                                                                            t=t_init, data=sitesdiff,
                                                                            nsample=num_sample)

        # Calculate values for burn_in
        burn_in_tau = get_burn_in_duration(s_tau)
        burn_in_theta = get_burn_in_duration(s_theta)
        create_joint_distribution(s_tau, s_theta)
        # Plot results, along with acceptance ratios.
        # plot_acceptance_ratios(p_accept_tau, p_accept_theta, p_accept_t, s)
        # plot_solution_distibution(s_tau, s_theta, s, burn_in_tau, burn_in_theta)
        #
        # # Get parameters to save
        # e_tau, e_theta = get_efficiency(s_tau, s_theta)
        # tau_ci = stats.norm(*stats.norm.fit(s_tau)).interval(0.95)
        # theta_ci = stats.norm(*stats.norm.fit(s_theta)).interval(0.95)

        # Add saved parameters to results
        # search_results.loc[i] = [0.01*s, 0.01*s, 0.002*s, e_tau[0], e_theta[0],
        #                          np.mean(p_accept_tau[1:]),  np.mean(p_accept_theta[1:]), np.mean(p_accept_t[1:]),
        #                          np.mean(s_tau[burn_in_tau:]), np.mean(s_theta[burn_in_tau:]), tau_ci[0], tau_ci[1],
        #                          theta_ci[0], theta_ci[1]]

    search_results.to_csv("./scale_search_results2.csv")
    return search_results


# Set random seed for reproducibility
random.seed(25)

# Run scale search
# df = scale_search([0.1, 0.08, 0.06], 100000)

df = scale_search([2, 1, 0.5, 0.4, 0.3, 0.2, 0.1, 0.08, 0.06], 100000)
# df = scale_search([0.2, 0.1, 0.08, 0.06], 10000)


# Run single search
# sitesdiff = load_data()
# t_init = [0.001 for i in range(1000)]
# s=0.1
# s_tau, s_theta, p_accept_tau, p_accept_theta, p_accept_t = run_mcmc(w_tau=0.01*s, w_theta=0.01*s, w_t=0.002*s,
#                                                                             tau=0.01, theta=0.001,
#                                                                             t=t_init, data=sitesdiff,
#                                                                             nsample=10000)
# plot_acceptance_ratios(p_accept_tau, p_accept_theta, p_accept_t, s)
# burn_in_tau = get_burn_in_duration(s_tau)
# burn_in_theta = get_burn_in_duration(s_theta)
# plot_solution_distibution(s_tau, s_theta, s, burn_in_tau, burn_in_theta)
