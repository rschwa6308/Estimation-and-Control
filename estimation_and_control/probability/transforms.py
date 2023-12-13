import numpy as np
from typing import Callable, Optional, Tuple

from estimation_and_control.helpers import interval_overlap
from .distributions import ProbabilityDistribution, GaussianDistribution, HistogramDistribution


"""
An 'uncertainty transform' models the affect of a function applied to a value that is
known with some uncertainty. Such a transform generally takes as input
 - the function being applied to the domain $f(X): R^n -> R^m$
 - the probability distribution representing our prior on the value $P(X): R^n -> R$
and returns
 - the approximate posterior belief on the value: $P(Y|X): R^n -> R$
 - potentially some additional information on the joint distribution $P(X,Y)$
"""


from .sigma_points import SigmaPointSelector


def unscented_transform(func: Callable, mean_x, cov_xx, sigma_point_selector: SigmaPointSelector, return_sigma_points=False) -> Tuple:
    """
    Given a function f(X) = Y together with the mean and covariance on X,
    apply the unscented transform to obtain the mean on Y, the covariance on Y, and the cross-covariance on X and Y
    """

    # select sigma points in accordance with prior
    sigma_points, weights_mean, weights_cov = sigma_point_selector.select_sigma_points(mean_x, cov_xx)

    # pass sigma points through the function
    sigma_points_transformed = []
    for i in range(sigma_points.shape[1]):
        point = sigma_points[:,[i]]
        x_prime = func(point)
        sigma_points_transformed.append(x_prime)
    
    sigma_points_transformed = np.hstack(sigma_points_transformed)

    # fit a gaussian to weighted transformed sigma points
    mean_y = np.sum(weights_mean * sigma_points_transformed, axis=1, keepdims=True)
    cov_yy = weights_cov * (sigma_points_transformed - mean_y) @ (sigma_points_transformed - mean_y).T
    cov_xy = weights_cov * (sigma_points - mean_x) @ (sigma_points_transformed - mean_y).T

    if return_sigma_points:
        return mean_y, cov_yy, cov_xy, sigma_points, sigma_points_transformed
    else:
        return mean_y, cov_yy, cov_xy



def histogram_transform(func: Callable, pdf: HistogramDistribution, output_pdf: Optional[HistogramDistribution] = None):
    "Explicit mass-transport transform from one HistogramDistribution to another"

    if pdf.dim > 1:
        raise NotImplementedError("Histogram transform for dim > 1 not yet supported.")

    # apply function to the entire domain
    func_values = func(pdf.domain)
    func_values_min, func_values_max = np.min(func_values), np.max(func_values)

    if output_pdf is None:
        # define the output domain to have sample density (approximately) equal to input domain
        # TODO: modify this behavior with a flag
        input_domain_width = pdf.domain[-1,0] - pdf.domain[0,0]
        output_domain_width = func_values_max - func_values_min
        domain_width_ratio = output_domain_width / input_domain_width
        
        output_pdf = HistogramDistribution(
            [(func_values_min, func_values_max)],
            np.round(pdf.bin_counts * domain_width_ratio).astype(int),
            None
        )
    else:
        # verify that function output fits within user-provided domain
        if func_values_min < output_pdf.domain_bounds[0,0] or func_values_min > output_pdf.domain_bounds[0,1]:
            raise ValueError("The domain of user-provided `output_pdf` is not large enough to accommodate the image of the input domain.")
            # TODO: offer auto-expand mode

    # set output pmf to 0, then accumulate mass
    output_pdf.pmf_values *= 0

    # iterate over input domain bins - TODO: vectorize
    for i in range(pdf.bin_counts[0]):
        mass = pdf.pmf_values[i]                        # mass in source bin [i, i+1]
        image = (func_values[i], func_values[i+1])      # mass transport destination

        # enforce image[0] <= image[1]
        image = (min(image), max(image))

        # volume of mass destination
        image_volume = image[1] - image[0]
        if image_volume == 0:
            print("WARNING: func is not locally invertible at current discretization! Some mass will be lost...")
            continue

        # identify the indices of the bins corresponding to the image
        image_bins = (
            output_pdf.get_bin_index(image[0]),
            output_pdf.get_bin_index(image[1])
        )

        # Distribute mass across output bins intersecting the destination region

        # The image is a polytope with 2*input_dim vertices,
        # (topologically a hypercube).
        # - in 1D: interval
        # - in 2D: quadrilateral
        # - in 3D: cube
        # - in 4D: hypercube
        # Note: image will be degenerate if output_dim < input_dim

        # Use a rasterization scheme

        # form the bounding box of the image (in bin space)
        image_bins_bbox = [
            np.floor(image_bins[0]).astype(int),
            np.ceil(image_bins[1]).astype(int),
        ]

        # TODO: decide on correct endpoint clipping
        image_bins_bbox[1] = np.clip(image_bins_bbox[1], 0, output_pdf.bin_counts[0]-1)

        # iterate over bins in bounding box
        for bin in range(image_bins_bbox[0][0], image_bins_bbox[1][0]):
            # compute volume of intersection between bin and image
            intersection_volume = interval_overlap(
                (output_pdf.domain[bin], output_pdf.domain[bin+1]),
                image
            )

            # compute portion of mass delivered to intersection
            mass_fraction = intersection_volume / image_volume

            # deliver mass
            output_pdf.pmf_values[bin] += mass * mass_fraction
    
    total_prob = np.sum(output_pdf.pmf_values)
    if not np.isclose(total_prob, 1.0, rtol=1e-4):
        print(f"WARNING: some mass was lost/gained during transform. (np.sum(output_pdf.pmf_values) == {total_prob:0.5f})")

    return output_pdf


