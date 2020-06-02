from typing import Optional

import numpy as np
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from rankcentrality.types import Scores, Comparisons, ComparisonResults, Matrix
import rankcentrality.internal as internal


class BTLMLE:
    def __init__(
        self, n_items: int, comps: Comparisons, comp_results: ComparisonResults
    ):
        """Use the BTL maximum liklihood to find a ranking.
        
        This class uses the reduction of BTL-MLE to logistic regression. 
        
        Note: The implementation of logistic regression used requires that
        there be a non-zero regularizer. The effect of this regularizer can
        be minimized by setting the keywork argument `C=` to a large positive
        number in the `fit_and_score` function.

        Args:
            n_items: The number of items being compared.
            comps: The pairs of comparisons made.
            comp_results: The results of pairwise comparisons.
        """
        n_comps = len(comps)
        comp_inds = np.arange(n_comps)
        features = np.zeros((n_comps, n_items))
        features[comp_inds, comps[comp_inds, comp_results]] = 1
        features[comp_inds, comps[comp_inds, 1 - comp_results]] = -1
        y = np.ones(n_comps)
        # sklearn needs to see two classes so invert half
        mask = np.linspace(0, 1, n_comps).round().astype(bool)
        y[mask] = -y[mask]
        features[mask, :] = -features[mask, :]
        self.features = features
        self.y = y

    def run(self, **logreg_kwargs) -> Scores:
        """Runs BTE-MLE using Logistic Regression from Scikit-Learn.

        Args:
            **logreg_kwargs: keyword arguments to pass to Scikit-Learn. An
                important example is `C`: larger values of `C` correspond to
                less regularization. For more options, see documentation
                online for `sklearn.linear_model.LogisticRegression`.
        """
        params = dict(fit_intercept=False, solver="liblinear")
        params.update(logreg_kwargs)
        lr = LogisticRegression(**params)
        lr.fit(self.features, self.y)
        return np.exp(lr.coef_.flatten())


class RankCentrality:
    def __init__(
        self, n_items: int, comps: Comparisons, comp_results: ComparisonResults
    ):
        self.n_items = n_items
        self.n_comps = len(comps)
        self.Q_hat = internal.get_transition_matrix(n_items, comps, comp_results)

    def run(self, skip_check: bool = False) -> Scores:
        """Runs (unregularized) RankCentrality.
        
        This method computes the stationary distribution of the Markov chain
        Q_hat, which was pre-computed in the constructor.

        Args:
            skip_check: If set to True, this function will not check if the
                corresponding Markov chain is ergodic. This can speed up the
                computation but may lead to weird results if the matrix is
                not ergodic (for example, the resulting vector might have
                negative entries).
        """
        return internal.get_stationary_distribution(self.Q_hat, skip_check)

    def run_regularized(self, lam: float):
        """Runs RankCentrality with lambda-regularization.

        Args:
            lam: The value of lambda used to regularize RankCentrality (see
            paper for more details).
        """
        assert 0 < lam < 1, ValueError(
            f"Lambda must be between 0 and 1 (exclusive). Got {lam}"
        )
        D_lamba = internal.d_lam(self.n_items, lam)
        # This is guaranteed to be ergodic, so skip the ergodicity check.
        return internal.get_stationary_distribution(
            self.Q_hat @ D_lamba, skip_check=True
        )

    def run_regularized_decayed(self, eta: float):
        """Runs RankCentrality with decayed lambda-regularization.
        
        As discussed in the paper, by decaying the regularization parameter
        lambda as O(m^(-1/2)), were m is the number of comparisons, the
        predictor becomes asymptotically unbiased. In other words, as the
        number of comparisons goes to infinity, the expected bias goes to
        zero.

        Args:
            eta: The coefficient in lambda = eta * m^(-1/2) used to
                regularize RankCentrality.
        """
        return self.run_regularized(eta * (self.n_comps ** -1 / 2))

    def get_diffusion_matrix(
        self,
        item_features: Matrix,
        kernel_width: float,
        threshold: float = 0,
        metric: str = "euclidean",
        lam: Optional[float] = None,
    ) -> Matrix:
        """Gets the diffusion matrix (can be cached for performance).

        Args:
            item_features: A matrix of shape (n_items, n_dimensions)
            kernel_width: The kernel width sigma used to compute affinity.
            threshold: amount by which to soft-threshold entries in the
                afinity matrix.
            metric: A distance metric from which to compute affinity. Any
                value supported by scipy's pairwise distance methods
                (scipy.spatial.distance.pdist) is acceptable.
            lam: (optional) A value of lambda to further regularize the
                RankCentrality process. It is often helpful to set this to a
                small positive value when the number of comparisons is
                extremely low.
        """
        D_diff = internal.get_affinity_matrix(
            item_features, kernel_width, threshold, metric, True
        )
        if lam:
            D_diff = D_diff @ internal.d_lam(self.n_items, lam)
        return D_diff

    def run_diffused(
        self,
        item_features: Matrix,
        kernel_width: float,
        threshold: float = 0,
        metric: str = "euclidean",
        lam: Optional[float] = None,
    ) -> Scores:
        """Runs Diffusion-based Regularized RankCentrality.

        Args:
            item_features: A matrix of shape (n_items, n_dimensions)
            kernel_width: The kernel width sigma used to compute affinity.
            threshold: amount by which to soft-threshold entries in the
                afinity matrix.
            metric: A distance metric from which to compute affinity. Any
                value supported by scipy's pairwise distance methods
                (scipy.spatial.distance.pdist) is acceptable.
            lam: (optional) A value of lambda to further regularize the
                RankCentrality process. It is often helpful to set this to a
                small positive value when the number of comparisons is
                extremely low.
        """
        D_diff = self.get_diffusion_matrix(
            item_features, kernel_width, threshold, metric, lam
        )
        return internal.get_stationary_distribution(self.Q_hat @ D_diff)

    def run_diffused_decayed(
        self,
        item_features: Matrix,
        kernel_width: float,
        threshold: float = 0,
        metric: str = "euclidean",
        lam: Optional[float] = None,
    ) -> Scores:
        """Runs Diffusion-based (decayed) Regularized RankCentrality.

        The decayed regularization accounts for the asymptotic bias.

        Args:
            item_features: A matrix of shape (n_items, n_dimensions)
            kernel_width: The kernel width sigma used to compute affinity.
            threshold: amount by which to soft-threshold entries in the
                afinity matrix.
            metric: A distance metric from which to compute affinity. Any
                value supported by scipy's pairwise distance methods
                (scipy.spatial.distance.pdist) is acceptable.
            lam: (optional) A value of lambda to further regularize the
                RankCentrality process. It is often helpful to set this to a
                small positive value when the number of comparisons is
                extremely low.
        """
        D_diff = self.get_diffusion_matrix(
            item_features, kernel_width, threshold, metric, lam
        )
        D_decayed = (1 / np.sqrt(self.n_comps)) * D_diff + (
            1 - 1 / np.sqrt(self.n_comps)
        ) * np.eye(self.n_items)
        return internal.get_stationary_distribution(self.Q_hat @ D_decayed)

    def run_custom_regularizer(
        self, regularizer: Matrix, skip_check: bool = False
    ) -> Scores:
        """Runs RankCentrality with a custom regularizer matrix.

        Args:
            regularizer: A row stochastic matrix of shape (n_items, n_items).
        """
        return internal.get_stationary_distribution(
            self.Q_hat @ regularizer, skip_check
        )


class RankSVM:
    def __init__(
        self,
        n_items: int,
        comps: Comparisons,
        comp_results: ComparisonResults,
        item_features: Matrix,
    ):
        """Fits a RankSVM model on the given pairwise comparisons.
        
        RankSVM uses item features to compute a ranking. Therefore, the
        performance of the algorithm is determined solely by how well a
        linear function of the item features describes the relative scores of
        the items.
        
        Note: the scores returned by RankSVM are not BTL scores. In
        particular, these values do not estimate the probability of outcomes
        of pairwise comparisons.
        """
        self.item_features = item_features
        self.comps = comps
        self.comp_results = comp_results
        self.svc = LinearSVC(fit_intercept=False, max_iter=10000)
        self.pw_transform = None
        self.target_label = None

    def _pairwise_transform(self, features):
        """Performs pairwise transformation.

        Args:
            features: Features to use.
        """
        self.pw_transform = (
            features[self.comps[:, 0], :] - features[self.comps[:, 1], :]
        )
        self.target_label = np.array([1, -1])[self.comp_results]

    def run(self):
        """Runs the RankSVM algorithm."""
        self._pairwise_transform(self.item_features)
        self.svc.fit(self.pw_transform, self.target_label)
        scores = self.svc.decision_function(self.item_features)
        return scores.flatten()

    def run_random_features(self):
        """Runs the RankSVM algorithm with a random features transformation."""
        rbf = RBFSampler()
        features = rbf.fit_transform(self.item_features)
        self._pairwise_transform(features)
        self.svc.fit(self.pw_transform, self.target_label)
        scores = self.svc.decision_function(features)
        return scores.flatten()
