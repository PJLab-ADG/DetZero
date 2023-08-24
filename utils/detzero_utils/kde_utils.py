from torch import distributions


class KernelDensityEstimation:
    def __init__(self, bandwidth):
        self.bandwidth = bandwidth

    def kde(self, input):
        """
        Args:
            input: (N, E, nsample, D)
        """
        raise NotImplementedError

    def score_samples(self, samples, balls_idx, est_points):
        """
        Args:
            samples: (N, nsample, 3) tensor
            balls_idx: (N, nsample)
            est_points: (N, E, 3) tensor
        Returns:
            kde_output: (N, E)
        """
        assert len(samples.shape) == 3
        assert len(balls_idx.shape) == 2
        assert len(est_points.shape) == 3

        # Add dimensions
        samples = samples.unsqueeze(-3) # (N, 1, nsample, 3)
        est_points = est_points.unsqueeze(-2) # (N, E, 1, 3)

        kde_input = (est_points - samples) / self.bandwidth # (N, E, nsample, 3)

        kernel_output = self.kde(kde_input) # (N, E, nsample)

        # Mask out unused elements
        kde_mask = balls_idx.unsqueeze(-2).repeat(1, kernel_output.shape[-2], 1).bool()
        kernel_output[~kde_mask] = 0.

        balls_num = balls_idx.sum(-1).unsqueeze(-1)
        empty_balls_mask = (balls_num == 0).repeat(1, kernel_output.shape[-2])

        # Add normalization factor
        kde_output = kernel_output.sum(-1) / (self.bandwidth ** samples.shape[-1] * balls_num) # (N, E)
        kde_output[empty_balls_mask] = 0

        return kde_output


class GaussianKernelDensityEstimation(KernelDensityEstimation):
    def __init__(self, bandwidth, loc=0., scale=1.):
        super().__init__(bandwidth=bandwidth)
        self.kde_func = distributions.Normal(loc=loc, scale=scale)

    def kde(self, input):
        """
        Args:
            input: (N, E, nsample, D)
        """
        return self.kde_func.log_prob(input).sum(-1).exp()
