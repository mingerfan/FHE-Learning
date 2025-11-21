import numpy as np
from numpy.polynomial import Polynomial

class SimpleEncryption:
    def __init__(self, q: int, M: int):
        self.qL = q  # 模数列表
        self.M = M  # 调制阶数
        self.N = M // 2
        self.sigma = 3.2  # 高斯分布的标准差
        self.mod = Polynomial([1] + [0]*(self.N-1) + [1])  # X^N + 1

    def ZO(self, rho: float) -> Polynomial:
        return Polynomial(np.random.choice([-1, 0, 1], size=self.N, p=[rho/2, 1-rho, rho/2]))
    
    def HWT(self, h: int) -> Polynomial:
        """生成HWT分布的向量"""

        """
        indices 是一个长度为 h 的整数数组，表示要赋值的位置。
        signs 也是一个长度为 h 的数组，表示每个位置要赋的值(-1 或 1)。
        NumPy 支持“花式索引(fancy indexing)”和“广播赋值”，即当你用一个数组 indices 选取多个位置时，可以直接用另一个同长度的数组 signs 赋值, NumPy 会自动一一对应赋值。
        """
        vec = np.zeros(self.N, dtype=int)
        indices = np.random.choice(self.N, size=h, replace=False)
        signs = np.random.choice([-1, 1], size=h)
        vec[indices] = signs
        return Polynomial(vec)
    
    def DG(self, sigma: float) -> Polynomial:
        """生成高斯分布的向量"""
        samples = np.random.normal(0, sigma, size=self.N)
        dicrete_samples = np.round(samples).astype(int)
        return Polynomial(dicrete_samples)
    
    def sample_RqL(self) -> Polynomial:
        """生成均匀分布的多项式系数向量"""
        return Polynomial(np.random.randint(0, self.qL, size=self.N))
    
    def modular_reduce(self, poly: Polynomial) -> Polynomial:
        """对多项式系数进行模M约减"""
        reduced = poly % self.mod
        print(f"Reduced Polynomial Coefficients (before mod qL): {reduced.coef}")
        
        # 处理复数警告，只取实部
        coeffs = reduced.coef
        if np.iscomplexobj(coeffs):
            coeffs = np.real(coeffs)

        reduced_coeffs = np.array([np.round(coeff).astype(int) % self.qL for coeff in coeffs])
        return Polynomial(reduced_coeffs)

    


if __name__ == "__main__":
    enc = SimpleEncryption(q=257, M=16)
    sk = enc.HWT(h=4)
    print("Private Key (s):", sk)
    v = enc.ZO(rho=0.5)
    print("Random Vector (v):", v)
    e = enc.DG(sigma=3.2)
    print("Error Vector (e):", e)