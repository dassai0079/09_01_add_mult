import unittest
import numpy as np
from spline import Spline3Interpolator


class TestSpline3Interpolator(unittest.TestCase):

    def test_interpolator_basic_three_points(self):
        # 3点なら常に補間可能
        points = [[0, 0], [1, 1], [2, 0]]
        interpolator = Spline3Interpolator(points)
        x, y = interpolator.compute()
        self.assertIsInstance(x, np.ndarray)
        self.assertIsInstance(y, np.ndarray)
        self.assertGreater(len(x), 0)
        self.assertGreater(len(y), 0)

    def test_interpolator_point_ordering(self):
        # 入力順が逆でも補間結果は問題なく出ることを確認
        points = [[2, 0], [0, 0], [1, 1]]
        interpolator = Spline3Interpolator(points)
        x, y = interpolator.compute()
        self.assertTrue(np.all(np.diff(x) > 0))  # xは昇順であるべき

    def test_insufficient_points(self):
        # 1点だけなら補間不能（空配列を返す）
        points = [[0, 0]]
        interpolator = Spline3Interpolator(points)
        x, y = interpolator.compute()
        self.assertEqual(len(x), 0)
        self.assertEqual(len(y), 0)

    def test_param_length(self):
        # 補間後のパラメータ長チェック（4nの係数がある）
        points = [[0, 0], [1, 1], [2, 0]]
        interpolator = Spline3Interpolator(points)
        interpolator.compute()
        self.assertEqual(len(interpolator.params), 4 * (len(points) - 1))

    def test_monotonicity_of_x(self):
        # 出力xが常に昇順であることを確認
        points = [[-1, 0], [0, 1], [1, 0]]
        interpolator = Spline3Interpolator(points)
        x, _ = interpolator.compute()
        self.assertTrue(np.all(np.diff(x) > 0))


if __name__ == "__main__":
    unittest.main()
