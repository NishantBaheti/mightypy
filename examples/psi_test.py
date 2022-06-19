import numpy as np
from mightypy.stats import population_stability_index

np.random.seed(10)

expected_continuous = np.random.normal(size=(500,))
actual_continuous = np.random.normal(size=(500,))
psi_df = population_stability_index(expected_continuous, actual_continuous, data_type='continuous')
print(psi_df.psi.sum())


expected_discrete = np.random.randint(0,10, size=(500,))
actual_discrete = np.random.randint(0,10, size=(500,))
psi_df = population_stability_index(expected_discrete, actual_discrete, data_type='discrete')
print(psi_df.psi.sum())


expected_continuous = np.random.normal(size=(500,))
actual_continuous = np.random.normal(size=(500,))
psi_df = population_stability_index(expected_continuous, actual_continuous, data_type='some_random_number')
print(psi_df.psi.sum())