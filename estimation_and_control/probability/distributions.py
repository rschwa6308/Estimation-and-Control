from typing import List
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from estimation_and_control.helpers import lerp, cartesian_product
# from probability.visualization import plot_covariance_ellipse, plot_pdf_values
from mpl_toolkits.axes_grid1 import make_axes_locatable



class ProbabilityDistribution:
    type = "GENERIC"

    def __init__(self, dim: int):
        self.dim = dim

    def pdf(self, x) -> float:
        pass

    def get_mean(self):
        pass

    def get_covariance(self):
        pass


class ParametricDistribution(ProbabilityDistribution):
    pass


class GaussianDistribution(ParametricDistribution):
    type = "GaussianDistribution"

    def __init__(self, mean, covariance):
        dim = covariance.shape[0]
        super().__init__(dim)

        self.mean = mean
        self.covariance = covariance

    def pdf(self, x):
        x_flat = x.reshape(-1, self.dim)

        cov_det = np.linalg.det(self.covariance)
        cov_inv = np.linalg.inv(self.covariance)
        
        eta = 1 / np.sqrt((2*np.pi)**self.dim * cov_det)
        vals = eta * np.exp(-0.5 * np.sum((x_flat - self.mean) @ cov_inv * (x_flat - self.mean), axis=1))

        vals = vals.reshape(x.shape[:-1])
        return vals
    
    def __repr__(self):
        return f"GaussianDistribution(mean={self.mean}, covariance={self.covariance})"
    
    def get_mean(self):
        return self.mean
    
    def get_covariance(self):
        return self.covariance


class HistogramDistribution(ProbabilityDistribution):
    """
    A direct numerical representation of a PDF, formulated as a D-dimensional histogram over a rectangular region with uniform bins.

     - `domain_bounds`: a list of D 2-tuples representing the lower and upper bounds on each dimension of the domain
     - `bin_counts`: an array of integers representing the number of bins to discretize each dimension into
     
    The `domain` itself is constructed internally as the cartesian product of `[np.linspace(*domain_bounds[i], bin_counts[i]+1) for i in range(D)]`

     - `pmf_values[i,...,k]` represents the total probability mass in the rectangular region between
       `domain[i,...,k]` and `domain[i+1,...,k+1]`, and thus has shape one less than `domain` along each dimension:
       `(domain.shape[0]-1, ..., domain.shape[-1]-1)`
    
    If `pmf_values=None`, the uniform distribution will be assumed.

    Example (1D): the uniform distribution on (0, 50):
    ```
    >>> domain_bounds = [(0.0, 5.0)]
    >>> bin_counts = [5]
    >>> pmf_values =  np.array([0.2,  0.2,  0.2,  0.2,  0.2])
    >>> HistogramDistribution(domain_bounds, bin_counts, pmf_values)
    ```

    Example (2D): the uniform distribution on (-1, +1) x (-5, 5):
    ```
    >>> domain_bounds = [(-1.0, +1.0), (-5.0, 5.0)]
    >>> bin_counts = [200, 1000]      # yields square bins
    >>> pmf_values = np.ones((200, 1000))
    >>> pmf_values /= np.sum(pmf_values)
    >>> HistogramDistribution(domain_bounds, bin_counts, pmf_values)
    ```
    """

    type = "HistogramDistribution"

    def __init__(self, domain_bounds, bin_counts, pmf_values):
        # Set up domain
        assert(len(domain_bounds) == len(bin_counts))
        super().__init__(len(domain_bounds))

        self.domain_bounds = np.array(domain_bounds)
        self.bin_counts = np.array(bin_counts)

        self.domain = cartesian_product([
            np.linspace(*domain_bounds[i], bin_counts[i]+1)
            for i in range(self.dim)
        ])

        # pre-compute some useful information about bin geometry
        self.steps = (self.domain_bounds[:,1] - self.domain_bounds[:,0]) / self.bin_counts
        self.bin_volume = np.prod(self.steps)

        self.bin_lowers = self.domain[(np.s_[:-1],) * self.dim]
        self.bin_uppers = self.domain[(np.s_[1:],) * self.dim]
        self.bin_midpoints = self.bin_lowers + self.steps/2

        # Set pmf_values
        if pmf_values is None:  # default to uniform distribution
            pmf_values = np.ones(self.bin_midpoints.shape[:-1])
            pmf_values /= np.sum(pmf_values)

        assert(all(domain_dim == pmf_values_dim for domain_dim, pmf_values_dim in zip(bin_counts, pmf_values.shape)))
        assert(np.isclose(np.sum(pmf_values), 1.0))

        self.pmf_values = pmf_values

    @property
    def pdf_values(self):
        return self.pmf_values / self.bin_volume

    @pdf_values.setter
    def pdf_values(self, values):
        self.pmf_values = values * self.bin_volume

    def get_bin_index(self, x):
        "Get the fractional index of the bin containing point x"
        return (x - self.domain_bounds[:,0]) / self.steps

    def pdf(self, x, interp=False):
        x_flat = x.reshape(-1, self.dim)

        # mask out query points that are outside the domain - they will be assigned value 0
        mask = np.all((x_flat >= self.domain_bounds[:,0]) & (x_flat < self.domain_bounds[:,1]), axis=1)

        index = self.get_bin_index(x_flat)
        # index = (x_flat - self.domain_bounds[:,0]) / self.steps
        # index -= 0.5        # PDF samples at bin midpoint!

        if not interp:
            # find the bin the query point belongs to
            index_bin = np.floor(index).astype(int)

            # return 0 if point is outside domain
            vals = np.zeros((x_flat.shape[0]))
            vals[mask] = np.take(self.pdf_values,
                np.ravel_multi_index(index_bin[mask].T, self.pdf_values.shape)
            )
        
        else:
            raise NotImplementedError()
            # index_low, index_high = np.floor(index).astype(int), np.ceil(index).astype(int)

            # index_low = np.clip(index_low, 0, None)
            # index_high = np.clip(index_high, None, self.bin_counts-1)
            # print(index)
            # print(index_low)
            # print(index_high)

            # t = index - index_low
            # print(self.pmf_values.shape)
            # values_low = np.take(self.pmf_values,
            #     np.ravel_multi_index(index_low.T, self.pmf_values.shape)
            # )
            # values_high = np.take(self.pmf_values,
            #     np.ravel_multi_index(index_high.T, self.pmf_values.shape)
            # )

            # print(values_low)
            # print(values_high)
            # print(t)

            # vals = lerp(values_low, values_high, t.reshape(-1, 1))

        vals = vals.reshape(x.shape[:-1])
        return vals

    def __repr__(self):
        return f"HistogramDistribution(domain_bounds=[{', '.join(map(str, self.domain_bounds))}], bin_counts={self.bin_counts})"

    def get_mean(self):
        values_flat = self.pmf_values.reshape(-1, 1)
        bin_midpoints_flat = self.bin_midpoints.reshape(-1, self.dim)
        return np.sum(values_flat * bin_midpoints_flat, axis=0)

    def get_covariance(self):
        mean = self.get_mean()
        values_flat = self.pmf_values.reshape(-1, 1)
        bin_midpoints_flat = self.bin_midpoints.reshape(-1, self.dim)
        return (bin_midpoints_flat - mean).T @ (values_flat * (bin_midpoints_flat - mean))




class MixtureDistribution(ProbabilityDistribution):
    def __init__(self, components: List[ProbabilityDistribution], weights):
        assert(all(c.dim == components[0].dim for c in components))
        assert(np.isclose(np.sum(weights), 1.0))
        assert(len(components) == len(weights))

        super().__init__(components[0].dim)

        self.components = components
        self.weights = np.array(weights)
    
    def pdf(self, x):
        assert(x.shape[-1] == self.dim)
        x_flat = x.reshape(-1, self.dim)

        component_values = np.array([c.pdf(x_flat) for c in self.components])
        vals = np.sum(self.weights.reshape(-1, 1) * component_values, axis=0)

        vals = vals.reshape(x.shape[:-1])
        return vals
    
    def get_mean(self):
        component_means = np.array([c.get_mean() for c in self.components])
        mean = np.sum(self.weights.reshape(-1, 1) * component_means, axis=0)
        return mean
    
    def get_covariance(self):
        # General derivation can be found here:
        # https://stats.stackexchange.com/questions/16608/what-is-the-variance-of-the-weighted-mixture-of-two-gaussians
        component_means = np.array([c.get_mean() for c in self.components])
        component_covs = np.array([c.get_covariance() for c in self.components])

        weights = self.weights.reshape(-1,1)
        mean = np.sum(weights * component_means, axis=0)

        cov = np.sum(weights.reshape(-1,1,1) * component_covs, axis=0) \
            + component_means.T @ (weights * component_means) \
            - mean.reshape(self.dim, 1) @ mean.reshape(1, self.dim)
        return cov




class ParticleDistribution(ProbabilityDistribution):
    type = "ParticleDistribution"
    pass    # TODO

