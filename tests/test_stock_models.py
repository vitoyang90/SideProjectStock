import unittest

from train_stock_models import select_tree_count


class StockModelTests(unittest.TestCase):
    def test_tree_count_uses_median_best_iteration_plus_one(self):
        self.assertEqual(select_tree_count([2, 7, 3, 5]), 5)

    def test_tree_count_rejects_empty_input(self):
        with self.assertRaises(ValueError):
            select_tree_count([])


if __name__ == "__main__":
    unittest.main()
