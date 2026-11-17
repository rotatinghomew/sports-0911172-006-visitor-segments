import unittest
from product.clustering import standardize, kmeans, silhouette, match_clusters


class ClusteringTests(unittest.TestCase):
    def test_standardize_constant_column_collapses_to_zero(self):
        scaled, means, stds = standardize([[1.0, 5.0], [3.0, 5.0]])
        self.assertEqual([row[1] for row in scaled], [0.0, 0.0])
        self.assertEqual(stds[1], 0.0)
        self.assertAlmostEqual(scaled[0][0], -1.0)
        self.assertAlmostEqual(scaled[1][0], 1.0)

    def test_kmeans_is_deterministic_and_separates_obvious_groups(self):
        points = [[0.0, 0.0], [0.1, 0.0], [0.0, 0.1], [9.0, 9.0], [9.1, 9.0], [9.0, 9.1]]
        first = kmeans(points, 2, seed=7)
        second = kmeans(points, 2, seed=7)
        self.assertEqual(first, second)
        inertia, assign, _ = first
        self.assertLess(inertia, 0.05)
        self.assertEqual(assign[:3], [assign[0]] * 3)
        self.assertEqual(assign[3:], [assign[3]] * 3)
        self.assertNotEqual(assign[0], assign[3])

    def test_silhouette_rewards_separated_groups(self):
        points = [[0.0], [0.1], [10.0], [10.1]]
        _, assign, _ = kmeans(points, 2, seed=3)
        self.assertGreater(silhouette(points, assign, 2), 0.9)
        self.assertEqual(silhouette(points, assign, 1), 0.0)

    def test_match_clusters_aligns_by_overlap_with_deterministic_ties(self):
        old = [{'a', 'b'}, {'c', 'd'}]
        new = [{'c', 'd', 'x'}, {'a', 'b'}]
        self.assertEqual(match_clusters(old, new), {0: 1, 1: 0})
        self.assertEqual(match_clusters([{'a'}, {'b'}], [{'a', 'b'}, set()]), {0: 0})


if __name__ == '__main__':
    unittest.main()
