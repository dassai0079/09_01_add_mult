import numpy as np
import matplotlib.pyplot as plt


class Spline3Interpolator:
    def __init__(self, points):
        self.points = sorted(points)
        self.n = len(self.points) - 1
        self.params = None

    def _build_equation(self):
        n = self.n
        A = np.zeros((4 * n, 4 * n))
        b = np.zeros(4 * n)

        # ステップ1: 通過条件
        for i in range(n):
            xi, xi1 = self.points[i][0], self.points[i + 1][0]
            for j in range(4):
                A[4 * i][4 * i + j] = xi ** (3 - j)
                A[4 * i + 1][4 * i + j] = xi1 ** (3 - j)
            b[4 * i] = self.points[i][1]
            b[4 * i + 1] = self.points[i + 1][1]

        # ステップ2・3: 接線・2階微分の連続性
        for i in range(1, n):
            xi = self.points[i][0]
            d1 = np.array([3 * xi ** 2, 2 * xi, 1, 0])
            d2 = np.array([6 * xi, 2, 0, 0])
            A[4 * i + 2][4 * (i - 1):4 * i] = d1
            A[4 * i + 2][4 * i:4 * (i + 1)] = -d1
            A[4 * i + 3][4 * (i - 1):4 * i] = d2
            A[4 * i + 3][4 * i:4 * (i + 1)] = -d2

        # ステップ4: 両端の自然境界条件
        A[2][0:4] = np.array([6 * self.points[0][0], 2, 0, 0])
        A[3][4 * (n - 1):4 * n] = np.array([6 * self.points[n][0], 2, 0, 0])

        return A, b

    def compute(self):
        if self.n <= 0:
            return np.array([]), np.array([])

        A, b = self._build_equation()
        self.params = np.linalg.solve(A, b)
        return self._interpolate_curve()

    def _interpolate_curve(self):
        x_all = np.empty(0)
        y_all = np.empty(0)
        for j in range(self.n):
            x_range = np.arange(self.points[j][0], self.points[j + 1][0], 0.01)
            coefs = self.params[4 * j:4 * j + 4]
            y_range = np.polyval(coefs, x_range)
            x_all = np.append(x_all, x_range)
            y_all = np.append(y_all, y_range)
        return x_all, y_all


def plot_curve(x, y, sample_points):
    plt.figure()
    plt.xlim([-1, 1])
    plt.ylim([-1, 1])
    plt.plot(x, y, label="Spline Curve")
    plt.scatter([p[0] for p in sample_points], [p[1] for p in sample_points], color='red', label="Sample Points")
    plt.legend()
    plt.grid(True)
    plt.show()


def main():
    # テスト用データ or 評価用データの選択
    sample_points = [
        [-0.8, -0.25],
        [-0.4, 0.3],
        [0.0, 0.3],
        [0.3, -0.25],
        [0.9, -0.2]
    ]

    interpolator = Spline3Interpolator(sample_points)
    x, y = interpolator.compute()
    plot_curve(x, y, sample_points)


if __name__ == "__main__":
    main()
