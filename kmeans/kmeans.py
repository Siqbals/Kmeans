import json
from dataclasses import dataclass
from typing import Any, Dict, Optional
from torchvision import datasets, transforms

import click
import numpy as np


@dataclass
class KMeansConfig:
    """This dataclass stores the KMeans clustering algorithm hyper-parameter.

    Parameter:
    * k: Number of centroids
    * max_iterations: Maximum number of improvement iterations.
    * epsilon: Distortion improvement threshold. If one iteration does not
        improve distortion by at least epsilon, then the algorithm stops early.
    """

    k: int
    max_iterations: int = 100
    epsilon: float = 1e-3

    def to_dict(self) -> Dict[str, Any]:
        return {
            "k": self.k,
            "max_iterations": self.max_iterations,
            "epsilon": self.epsilon,
        }


@dataclass
class KMeansResult:
    """This dataclass stored the clustering result of the KMeans clustering
    algorithm.

    The matrix `mu` has shape `[k, d]` for `k` clusters of dimension `d`. The
    `membership` matrix has shape `[n, k]` for `n` data points (images) and
    `k` clusters. This matrix can be stored using datatype float32.

    """

    config: KMeansConfig
    distortion: float
    mu: np.ndarray
    membership: np.ndarray

    def to_file(self, filename: str):
        kmeans_res_dict = {
            "config": self.config.to_dict(),
            "distortion": float(self.distortion),
            "mu": self.mu.tolist(),
            "membership": self.membership.tolist(),
        }
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(kmeans_res_dict, f)

    @staticmethod
    def from_file(filename: str) -> "KMeansResult":
        with open(filename, "r", encoding="utf-8") as f:
            res_dict = json.load(f)
        mu = np.array(res_dict["mu"], dtype=np.float32)
        membership = np.array(res_dict["membership"], dtype=np.float32)
        return KMeansResult(
            config=KMeansConfig(**res_dict["config"]),
            distortion=res_dict["distortion"],
            mu=mu,
            membership=membership,
        )


@dataclass
class MNIST7K:
    imgs: np.ndarray
    labels: np.ndarray

    @staticmethod
    def from_file() -> "MNIST7K":
        transform = transforms.ToTensor()

        # Download and load the MNIST training dataset
        dataset = datasets.MNIST(root="data", train=True, download=True, transform=transform)

        # Extract image data and labels
        imgs = np.array([img[0].squeeze(0).numpy() for img in dataset])  # [28, 28] images
        labels = np.array([img[1] for img in dataset])

        return MNIST7K(
            imgs=imgs.astype(np.float32),  # shape [n, 28, 28]
            labels=labels,
        )


def compute_distortion(
    x: np.ndarray,
    mu: np.ndarray,
    membership: np.ndarray,
) -> float:
    """Compute the distortion J (the loss objective that K-Means
    clustering optimizes.)

    Args:
        x (np.ndarray): Dataset matrix of shape `[n, d]`.
        mu (np.ndarray): Centroid matrix of shape `[k, d]`.
        membership (np.ndarray): Cluster membership bit matrix of shape
            `[n, k]`.

    Returns:
        float: Distortion value (value of the J loss objective).
    """
    # From the slides, J = ||X -ZM||^2
    #   X is our data, x
    #   Z is the membership array (one-hot encoding)
    #   M is our centroids

    # We want to first create zm, which is a [n,d] array where each row i is equal to the
    #   centroid which x_i belongs to.
    zm = np.dot(membership, mu)

    # Now we just need the componenent differences, then square and sum all
    square_diffs = np.power(x - zm, 2)
    distortion = np.sum(square_diffs)

    return distortion


class KMeans:
    def __init__(self, config: KMeansConfig):
        """K-Means clustering implementation.

        Args:
            config (KMeansConfig): KMeans clustering config.
        """
        self.config = config

    def init_centroids(self, x: np.ndarray) -> np.ndarray:
        """Initialize centroids by randomly picking points from the dataset.
        Start points can be picked uniformly at random.

        Args:
            x (np.ndarray): Dataset array `[n, d]`.

        Returns:
            np.ndarray: Centroid array of shape `[k, d]`.
        """

        # This picks k rows from all available points from x
        num_rows = x.shape[0]
        indexes = np.random.choice(num_rows, size=self.config.k, replace=False)
        centroids = x[indexes]

        return centroids

    def get_new_centroids(
        self,
        x: np.ndarray,
        membership: np.ndarray,
    ) -> np.ndarray:
        """Calculate and return centroids for points given membership data

        Args:
            x (np.ndarray): Dataset matrix of shape `[n, d]`.
            membership (np.ndarray): Cluster membership bit matrix of shape
                `[n, k]`.

        Returns:
            np.nparray: [k, d] array, which represents the positions of new centroids
        """
        d = x.shape[1]
        k = membership.shape[1]

        # Create a [k*n, d] array from x. Just k copies
        x_expanded = np.tile(x, (k, 1, 1)).reshape(-1, d)
        # Transpose membership to get [n, k], then reshape so [n*k, 1]
        mem_transposed = np.transpose(membership).reshape(-1, 1)

        # Broadcast to get [k, n, d] where:
        # In each block k, row i is non-zero if it belongs to centroid k
        temp = (mem_transposed * x_expanded).reshape(k, -1, d)
        totals = temp.sum(axis=1)

        # Now just need to scale by the number of elements with membership to each centroid
        counts = membership.sum(axis=0).reshape(-1, 1)

        centroids = totals / counts

        return centroids

    def get_membership(self, x: np.ndarray, mu: np.ndarray) -> np.ndarray:
        """Calculate the membership array based on the data points (x) and centroids (mu)

            Will compute distances from each row to each mu
            This will be [n, k ,d] array
            Sum over dimension d to get [n, k]
            Argmin each row so only 1 element in each row is 1, rest 0

        Args:
            x (np.ndarray) Dataset array of shape '[n, d]'
            mu (np.ndarray) Dataset array of shape '[k, d]'

        Returns:
            np.ndarray: Of shape '[n,k]'
        """

        # Get dimensions for expanding
        n, d = x.shape
        k = mu.shape[0]

        # Expand so the each point in x is repeated k times
        x_expand = np.tile(x, (1, 1, k)).reshape(n, k, d)
        # Repeat mu n times to match size
        mu_expand = np.tile(mu, (n, 1, 1))

        # Create distance matrix, where each [i,j] is the distance between point x_i and mu_j
        distances = np.sum(np.power(x_expand - mu_expand, 2), axis=2)

        # Get the argmin, since this will be the nearest centroid
        mins = np.argmin(distances, axis=1)

        # Index idenetity matrix by the minimums for each row to create membership matrix
        membership = np.identity(k)[mins]

        return membership

    def fit(
        self, x: np.ndarray, mu_init: Optional[np.ndarray] = None
    ) -> KMeansResult:
        """Runs the K-Means clustering procedure on the provided dataset.

        Args:
            x (np.array): Dataset array of shape `[n, d]`.
            mu_init (np.ndarray): Centroid initialization array of shape
                `[k, d]`.

        Returns:
            KMeansResult: K-Means clustering result.
        """
        if mu_init is None:
            mu_init = self.init_centroids(x)
        # Get initial memberships and distortion
        memberships = self.get_membership(x, mu_init)
        distortion = compute_distortion(x, mu_init, memberships)

        # Recalculate centroids and distortions
        for i in range(self.config.max_iterations):
            # Get new centroids
            mu = self.get_new_centroids(x, memberships)

            # Get new memberships
            memberships = self.get_membership(x, mu)

            # Get distortion
            new_distortion = compute_distortion(x, mu, memberships)
            print("Distortion after ", i + 1, " iterations is: ", distortion)

            # Compare to previous distortion. If not lower by at least epsilon, DONE
            improvement = distortion - new_distortion
            distortion = new_distortion
            if self.config.epsilon > improvement:
                # Didn't improve by enough, stopping early
                break

        # Build KMeansResult and return
        return KMeansResult(self.config, distortion, mu, memberships)

    def fit_repeats(
        self,
        x: np.ndarray,
        repeats: int,
    ) -> KMeansResult:
        """Repeats the K-Means cluster procedure by repeatedly calling the fit
        method above. The result with the lowest distortion is then returned.

        Args:
            x (np.array): Dataset array of shape `[n, d]`.
            repeats (int, optional): Number of repeats for which the algorithm is run.

        Returns:
            KMeansResult: Clustering result with the lowest distortion.
        """
        distortions = []
        results = []
        for i in range(repeats):
            print("Running fit: ", i + 1, " of ", repeats)
            this_result = self.fit(x)
            distortions.append(this_result.distortion)
            results.append(this_result)
            print(
                "Final distortion for repeat",
                i + 1,
                " of: ",
                this_result.distortion,
            )

        # Smallest final distortion
        best_index, _ = min(enumerate(distortions), key=lambda x: x[1])

        return results[best_index]


@click.command()
@click.option(
    "--filename",
    default="clustering.json",
    type=str,
    help="Cluster result filename.",
)
@click.option("--k", default=10, type=int, help="K parameter.")
@click.option(
    "--max-iterations",
    type=int,
    default=100,
    help="Maximum number of iterations.",
)
@click.option(
    "--epsilon",
    type=float,
    default=1e-3,
    help="Minimum distortion improvement per iteration.",
)
@click.option(
    "--repeats",
    type=int,
    default=5,
    help="Number of K-Means clustering repeats.",
)
def main(
    filename: str,
    k: int,
    max_iterations: int,
    epsilon: float,
    repeats: int,
):
    data = MNIST7K.from_file()
    print("Done loading data")
    kmeans = KMeans(
        KMeansConfig(
            k=k,
            max_iterations=max_iterations,
            epsilon=epsilon,
        )
    )
    kmeans_res = kmeans.fit_repeats(
        data.imgs.reshape(-1, 28 * 28),
        repeats=repeats,
    )
    kmeans_res.to_file(filename)


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
