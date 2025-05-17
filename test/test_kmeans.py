def test_kmeans_result_persistance():
    """Tests persistance of the KMeans result class."""
    import os

    import numpy as np

    from kmeans.kmeans import KMeansConfig, KMeansResult

    res = KMeansResult(
        config=KMeansConfig(k=10),
        distortion=1.0,
        mu=np.random.uniform((5, 10)),
        membership=np.random.uniform((50, 10)),
    )
    res.to_file("test_res.json")

    res_recon = KMeansResult.from_file("test_res.json")
    assert res.config.k == res_recon.config.k
    assert res.config.max_iterations == res_recon.config.max_iterations
    assert res.config.epsilon == res_recon.config.epsilon
    assert res.distortion == res_recon.distortion
    assert np.allclose(res.mu, res_recon.mu)
    assert np.allclose(res.membership, res_recon.membership)
    os.remove("test_res.json")


def test_distortion():
    """Tests correctness of distortion calculation"""
    import numpy as np
    from kmeans.kmeans import compute_distortion

    example_x = np.asarray(
        [[0, 0, 0, 0], [2, 2, 2, 2], [5, 1, 9, 4], [1, 2, 3, 1]]
    )
    centroids = np.asarray([[1, 1, 1, 1], [5, 1, 8, 5]])
    membership = np.asarray([[1, 0], [1, 0], [0, 1], [1, 0]])

    # Diffs should be
    # [[-1,-1,-1,-1],
    #  [1,1,1,1],
    #  [0,0,1,-1],
    #  [0,1,2,0]]

    # Then squared:
    # [[1,1,1,1],
    #  [1,1,1,1],
    #  [0,0,1,1],
    #  [0,1,4,0]]

    # Gives sum of 15
    dist = compute_distortion(example_x, centroids, membership)
    assert dist == float(15)


def test_distortion_floats():
    """Tests correctness of distortion calculation"""
    import numpy as np
    from kmeans.kmeans import compute_distortion

    example_x = np.asarray(
        [
            [1.3, 1.7, 1.3, 0.5],
            [2.3, 2.3, 2.1, 0.9],
            [5.3, 1.3, 9.3, 4.3],
            [1.12, 2.3, 3.0, 1.11],
        ]
    )
    centroids = np.asarray([[1.1, 1.1, 1.1, 1.1], [5.1, 1.1, 8.1, 5.1]])
    membership = np.asarray([[1, 0], [1, 0], [0, 1], [1, 0]])

    # Diffs should be
    # [[0.2, 0.6, 0.2, -0.6],
    #  [1.2, 1.2, 1.0, -0.2],
    #  [0.2, 0.2, 1.2, -0.8],
    #  [0.02, 1.2, 1.9, 0.01]]

    # Then squared:
    # [[0.04, 0.36, 0.04, 0.36], = 0.8
    #  [1.44, 1.44, 1, 0.04],    = 3.92
    #  [0.04, 0.04, 1.44, 0.64], = 2.16
    #  [0.0004, 1.44, 3.61, 0.0001]] = 5.0505

    #  0.8
    #  3.92
    #  2.16
    #  5.0505
    # = 11.9305

    dist = compute_distortion(example_x, centroids, membership)
    assert dist == float(11.9305)


def test_membership_floats():
    """Tests correctness of distortion calculation"""
    import numpy as np
    from kmeans.kmeans import KMeans, KMeansConfig, compute_distortion

    kmeans = KMeans(KMeansConfig(k=3))

    example_x = np.asarray(
        [
            [1.3, 1.7, 1.3, 0.5],
            [2.3, 2.3, 2.1, 0.9],
            [5.3, 1.3, 9.3, 4.3],
            [1.12, 2.3, 3.0, 1.11],
        ]
    )
    centroids = np.asarray([[1.1, 1.1, 1.1, 1.1], [5.1, 1.1, 8.1, 5.1]])
    expected_membership = np.asarray([[1, 0], [1, 0], [0, 1], [1, 0]])

    calc_membership = kmeans.get_membership(example_x, centroids)

    assert np.all(calc_membership == expected_membership)

    dist = compute_distortion(example_x, centroids, expected_membership)
    assert dist == float(11.9305)


def test_get_new_centroids():
    """Tests that the get_new_centroids function works as expected"""

    import numpy as np
    from kmeans.kmeans import KMeans, KMeansConfig

    kmeans = KMeans(KMeansConfig(k=3))

    membership = np.asarray(
        [
            [1, 0, 0],
            [0, 1, 0],
            [1, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 0],
            [0, 1, 0],
        ]
    )

    x = np.asarray(
        [
            [0, 0, 0],
            [3, 4, 3],
            [0, 0, 1],
            [9, 0, 9],
            [4, 4, 4],
            [10, 10, 11],
            [0, 1, 2],
            [5, 2, 3],
        ]
    )
    expected_centroids = np.asarray(
        [[0, 1 / 3, 1], [4, 10 / 3, 10 / 3], [9.5, 5, 10]]
    )

    calc_centroids = kmeans.get_new_centroids(x, membership)
    assert np.all(calc_centroids == expected_centroids)
