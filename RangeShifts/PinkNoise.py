import numpy as np
from scipy.stats import linregress, t
from logging import mc_logger


def noise_pink(nu, tmax, **kwargs):
    """
    Generate pink noise.

    Parameters:
    nu (float): Exponent controlling the frequency distribution.
    tmax (int): Maximum time point.
    **kwargs: Additional keyword arguments.
        beta (float, optional): Base for temporal spacing (default is 4).
        cutoffM (int, optional): Upper cutoff (default is 8).
        cutoffm (int, optional): Lower cutoff (default is 0).
        std (float, optional): Standard deviation of noise (default is 0.01).

    Returns:
    numpy.ndarray: An array containing pink noise data.
    """

    # Parameters
    beta = kwargs.get('beta', 4) # Base for temporal spacing
    cutoffM = kwargs.get('cutoffM', 8) # Upper cutoff
    cutoffm = kwargs.get('cutoffm', 0) # Lower cutoff
    std = kwargs.get('std', 0.01)

    # Define the temporal frequencies
    k = np.arange(cutoffm, cutoffM+1)

    # Compute the scaling factor for the temporal frequencies
    Dphi = np.log(beta)

    # Compute the temporal scales and weights
    tau = beta**k
    ro = np.exp(-1/tau)
    w = np.exp((nu-1)*k*Dphi)

    # Compute the weights for the autoregressive process
    W = w / np.sum(w)

    # Initialise arrays for the autoregressive process
    a = np.zeros((len(k), tmax))
    n = np.random.normal(0, std, (len(k), tmax))

    # Initialise the first time point
    a[:,0] = np.random.normal(0, std, size=len(k))

    for ti in range(1, tmax):
      a[:,ti] = ro*a[:,ti-1] + np.sqrt(1-ro**2)*n[:,ti-1]

    # Compute the noise signal as the weighted sum of the autoregressive process
    noise = np.sum(a * W.reshape((-1,1)), axis=0)

    return noise


def noise_white( tmax, **kwargs):
    """
    Generate white noise. Uses np.random.normal() directly. Implemented only
    for comparison.

    Parameters:
    tmax (int): Maximum time point.
    **kwargs: Additional keyword arguments.
        std (float, optional): Standard deviation of noise (default is 0.01).

    Returns:
    numpy.ndarray: An array containing white noise data.
    """
    # Parameters
    std = kwargs.get('std', 0.01)

    noise = np.random.normal(0, std, tmax)

    return noise




def MonteCarlo_significance(xy_array, MC_reps, noise_func, noise_kwargs,log_kwargs={}):

    """
    Run Monte Carlo simulations to check the significance of slopes.

    Parameters:
    xy_array (numpy.ndarray): A 2D array where the first column represents x data and the second column represents y data.
    MC_reps (int): Number of Monte Carlo repetitions.
    noise_func (function): A function to generate noise.
    noise_kwargs (dict): Keyword arguments for the noise function.

    Returns:
    list: A list containing the p-value, slope, and intercept.
    """
    ## logging _____________________________________| 
    mc_logger.log_function_call(function_name ="MonteCarlo_significance()",
                               args = {'MC_reps' : MC_reps,
                                       'noise_func':str(noise_func),
                                       'noise_kwargs':noise_kwargs},
                               **log_kwargs)
    ## =============================================|  

    x_array, y_array = xy_array

    # Fit a linear regression line to the data_points
    slope, intercept = np.polyfit(x_array, y_array,deg= 1)

    # Calculate the absolute value of the slope
    abs_slope = abs(slope)

    # Create an array of NaN values for storing noise slopes
    slopes = np.full(MC_reps-1, np.nan)


    # MonteCarlo Simulation
    # Simulate noise and fit linear regression lines for a given number of repetitions
    for i in range(MC_reps-1):

      # Define the noise function
      noise = noise_func(**noise_kwargs)

      # Scale the noise to have the same standard deviation as the original data
      scaled_noise = noise - np.mean(noise)
      scaled_noise *= np.std(y_array,ddof=1) / np.std(scaled_noise,ddof=1)
      scaled_noise += np.mean(y_array)

      # Calculate the the slope of the fitted line of noise
      slopes[i],_ = np.abs(np.polyfit(x_array, scaled_noise,deg= 1))

    # Calculate the proportion of times the slope of the simulated noise is
    # greater than the absolute value of the slope from the original data
    p_value = (sum(slopes > abs_slope) + 1) / (2 * MC_reps)

    ## logging _____________________________________|  
    estimations = []

    for i in range(100, len(slopes), 100):
        estimations.append((sum(slopes[:i] > abs_slope) + 1) / (2 * i))

    mc_logger.log_function_output(function_name = "MonteCarlo_significance()",
                                  output = {'p_value':p_value,
                                            'slope':slope,
                                            'intercept':intercept},
                                  estimation_values = estimations
                                 )

    ## =============================================|

    # Return the calculated proportion
    return [p_value,slope,intercept]


def white_fit_significance(xy_array):
    """
    Calculate significance by fitting a line to the data using scipy.stats.linregress.

    Parameters:
    xy_array (numpy.ndarray): A 2D array where the first column represents x data and the second column represents y data.

    Returns:
    float: The calculated p-value for the linear fit.
    """
    x_array, y_array = xy_array

    # Calculate linear regression
    slope, intercept, r_value, p_value, std_err = linregress(x_array, y_array)

    # Return p-value
    return p_value

'''
def white_fit_significance(xy_array):
    """
    Calculate significance by fitting a line to the data.

    Parameters:
    xy_array (numpy.ndarray): A 2D array where the first column represents x data and the second column represents y data.

    Returns:
    float: The calculated p-value for the linear fit.
    """
    x_array, y_array = xy_array

    # Linear Fit
    slope, intercept = np.polyfit(x_array, y_array, deg = 1)
    y_pred = slope * x_array + intercept

    # Calculate mse and ssx
    residuals = y_array - y_pred
    mse = np.sum(residuals ** 2) / (len(x_array) - 2)
    ssx = np.sum((x_array - np.mean(x_array)) ** 2)

    #p_value calculation
    t_value = slope / np.sqrt(mse / ssx)
    deg_freedom = len(y_array) - 2
    p_value = (1 - t.cdf(abs(t_value), deg_freedom)) * 2

    #return p_value
    return p_value
'''

def pink_fit_significance(data_points):
    """
    Calculate significance by fitting a line to the data (not yet implemented).

    Parameters:
    data_points (numpy.ndarray): Data points to analyze.

    Returns: p_value (as in white_fit_significance())

    None: The function is not yet implemented.
    Needs to be implemented using: https://arxiv.org/abs/1407.7760
    """
    pass


def MonteCarlo_compatibleTrends(xy_array, fitted_slope, noise_func, noise_kwargs, n_trends, max_iterations=1000):
    """
    Run Monte Carlo simulations to find compatible trends.

    Parameters:
    xy_array (numpy.ndarray): A 2D array where the first column represents x data and the second column represents y data.
    fitted_slope (float): The slope of the original data.
    noise_func (function): A function to generate noise.
    noise_kwargs (dict): Keyword arguments for the noise function.
    n_trends (int): Number of compatible trends to find.
    max_iterations (int, optional): Maximum number of iterations (default is 1000).

    Returns:
    tuple: A tuple containing slopes and intercepts of compatible trends.
    """

    x_array, y_array = xy_array
    slopes = np.full(n_trends, np.nan)
    intercepts = np.full(n_trends, np.nan)
    k = 0

    for _ in range(max_iterations):
        if k >= n_trends:
            break

        noise = noise_func(**noise_kwargs)
        scaled_noise = noise - np.mean(noise)
        scaled_noise *= np.std(y_array, ddof=1) / np.std(scaled_noise, ddof=1)
        scaled_noise += np.mean(y_array)

        slopes_cand, intercepts_cand = np.polyfit(x_array, scaled_noise, deg=1)

        if np.abs(slopes_cand) > np.abs(fitted_slope):
            slopes[k], intercepts[k] = slopes_cand, intercepts_cand
            k += 1

    return (slopes, intercepts)
