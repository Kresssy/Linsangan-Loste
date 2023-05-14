import pandas as pd
import numpy as np
from scipy.stats import norm, lognorm, t, expon, pareto, chi2
from scipy.stats import kstest, anderson

# Load the Bitcoin returns data from a CSV file or any other source
data = pd.read_csv('bitcoin_returns.csv')

# Assuming the returns are in a column named 'returns'
returns = data['returns']

# Fit different probability distributions to the data
distributions = {
  'Normal': norm,
  'Lognormal': lognorm,
  'Student\'s t': t,
  'Exponential': expon,
  'Pareto': pareto,
  'Chi-squared': chi2
}

best_fit = None
best_fit_name = ''
best_fit_params = ()
best_fit_score = np.inf

for name, distribution in distributions.items():
  # Fit the distribution to the data
  params = distribution.fit(returns)

# Calculate the goodness-of-fit test score
_, p_value = kstest(returns, distribution.cdf, args=params)
score = -np.log10(p_value)

# Update the best fit if the current distribution has a better score
if score < best_fit_score:
  best_fit = distribution
best_fit_name = name
best_fit_params = params
best_fit_score = score

# Perform the Anderson-Darling test for additional evaluation
ad_statistic, ad_critical_values, ad_significance_level = anderson(returns, dist=best_fit_name.lower())

# Print the best fit distribution and its parameters
print('Best Fit Distribution:', best_fit_name)
print('Parameters:', best_fit_params)
print('Anderson-Darling Test Statistic:', ad_statistic)
print('Critical Values:', ad_critical_values)
print('Significance Level:', ad_significance_level)