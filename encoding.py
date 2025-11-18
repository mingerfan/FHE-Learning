import numpy as np
from math import gcd

class SimpleEncoding:
    def __init__(self, M: int):
        self.M = M  # 调制阶数
        assert M & (M - 1) == 0 and M >= 2, "M must be a power of 2 and at least 2."

        self.N = M // 2
        self.roots = [np.exp(2j * np.pi * k / M) for k in range(M) if gcd(k, M) == 1] # 通过分园多项式结论得知：该式子得到X^N+1对应的所有单位根(M=2N)

        self.A = np.array([[root ** i for i in range(self.N)] for root in self.roots])  # 构造矩阵A
        self.AT = self.A.T
        self.AH = np.conj(self.AT)  # 共轭转置矩阵A^H

        self.orthogonal_basis = []
        for row in range(self.N):
            poly = np.polynomial.Polynomial(np.array([0 if i != row else 1 for i in range(self.N)]))
            self.orthogonal_basis.append(poly)

    def pi(self, z: np.ndarray) -> np.ndarray:
        return z[:self.N//2]
    
    def pi_inv(self, z_half: np.ndarray) -> np.ndarray:
        v_conj_rev = np.conj(z_half)[::-1]
        return np.concatenate([z_half, v_conj_rev])

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

    base1 = np.polynomial.Polynomial([1, 0, 0, 0])  # 多项式1
    base2 = np.polynomial.Polynomial([1, 0, 0, 0])  # 多项式X
    print(f"b1 dot b2 (多项式系数点积): {np.dot(base1.coef, base2.coef)}")
    v1 = encoder.sigma(base1)
    v2 = encoder.sigma(base2)
    print(f"b1 dot b2 (典范嵌入映射点积): {np.vdot(v1, v2)}\n")

    # 示例多项式：f(x) = 1 + 2x + 3x^2 + 4x^3
    poly = np.polynomial.Polynomial([1, 2, 3, 4])
    print("原始多项式系数:", poly.coef)

    # 典范嵌入映射
    vector = encoder.sigma(poly)
    print("映射后的向量表示:", vector)

    # 典范嵌入逆映射
    recovered_poly = encoder.sigma_inv(vector)
    print("恢复的多项式系数:", recovered_poly.coef)
    
