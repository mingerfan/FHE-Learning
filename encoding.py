import numpy as np
from math import gcd

class SimpleEncoding:
    def __init__(self, M: int):
        self.M = M  # 调制阶数
        assert M & (M - 1) == 0 and M >= 2, "M must be a power of 2 and at least 2."

        self.N = M // 2
        self.roots = [np.exp(2j * np.pi * k / M) for k in range(M) if gcd(k, M) == 1] # 通过分园多项式结论得知：该式子得到X^N+1对应的所有单位根(M=2N)

        self.A = np.array([[root ** i for i in range(self.N)] for root in self.roots])  # 构造矩阵A

    def sigma(self, poly: np.polynomial.Polynomial) -> np.ndarray:
        """典范嵌入映射：多项式系数 -> 向量表示"""
        return self.A @ poly.coef
    
    def sigma_inv(self, vector: np.ndarray) -> np.polynomial.Polynomial:
        """典范嵌入逆映射：向量表示 -> 多项式系数"""
        coeffs = np.linalg.solve(self.A, vector)
        return np.polynomial.Polynomial(coeffs)
    
    def modular_reduce(self, poly: np.polynomial.Polynomial) -> np.polynomial.Polynomial:
        """对多项式系数进行模M约减"""
        modulo = np.polynomial.Polynomial([1] + [0]*(self.N - 1) + [1])  # X^N + 1
        return poly % modulo
    
if __name__ == "__main__":
    M = 8
    encoder = SimpleEncoding(M)

    # 示例多项式：f(x) = 1 + 2x + 3x^2 + 4x^3
    poly = np.polynomial.Polynomial([1, 2, 3, 4])
    print("原始多项式系数:", poly.coef)

    # 典范嵌入映射
    vector = encoder.sigma(poly)
    print("映射后的向量表示:", vector)

    # 典范嵌入逆映射
    recovered_poly = encoder.sigma_inv(vector)
    print("恢复的多项式系数:", recovered_poly.coef)
    
