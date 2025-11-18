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
    
    def _project_to_basis(self, vector: np.ndarray) -> np.ndarray:
        """
        投射向量到正交基上
        
        Args:
            vector: 需要投射的向量
            
        Returns:
            在正交基下的系数
        """
        basis_vectors = [self.sigma(poly) for poly in self.orthogonal_basis]
        coeffs = []
        for vec in basis_vectors:
            coeff = np.vdot(vector, vec) / np.vdot(vec, vec)  # Hermitian内积
            coeffs.append(coeff)
        return np.array(coeffs)
    
    def _random_round(self, coeffs: np.ndarray) -> np.ndarray:
        """
        坐标随机舍入
        
        Args:
            coeffs: 需要舍入的系数（实数）
            
        Returns:
            舍入后的整数系数
        """
        coeffs_real = np.real(coeffs)
        f_part = coeffs_real - np.floor(coeffs_real)
        f_to_add = np.array([np.random.choice([-x, 1-x], 1, p=[1-x, x]) for x in f_part]).flatten()
        return (coeffs_real + f_to_add).astype(int)
    
    def _reproject_to_vector(self, coeffs: np.ndarray) -> np.ndarray:
        """
        使用基向量重新映射为向量
        
        Args:
            coeffs: 在正交基下的整数系数
            
        Returns:
            重新映射后的向量
        """
        basis_vectors = [self.sigma(poly) for poly in self.orthogonal_basis]
        basis_matrix = np.array(basis_vectors).T  # 列向量为基向量
        return basis_matrix @ coeffs
    
    def encode(self, z: np.ndarray, scale: float) -> np.polynomial.Polynomial:
        """
        编码过程：将复向量编码为多项式
        
        编码流程：
        1. z ∈ C^{N/2}
        2. π^{-1}(z) ∈ H
        3. Δ·π^{-1}(z)
        4. 投射到σ(R)中
        5. 随机舍入
        6. 重新映射为向量
        7. σ^{-1}编码为多项式
        
        Args:
            z: 待编码的复向量，长度为N/2
            scale: 缩放因子Δ
            
        Returns:
            编码后的多项式 m(X) ∈ R
        """
        # 1. 应用π逆映射：C^{N/2} -> H
        v_expanded = self.pi_inv(z)
        
        # 2. 缩放
        v_scaled = v_expanded * scale
        
        # 3. 投射到正交基
        coeffs = self._project_to_basis(v_scaled)
        
        # 4. 随机舍入
        coeffs_rounded = self._random_round(coeffs)
        
        # 5. 重新映射为向量
        v_rounded = self._reproject_to_vector(coeffs_rounded)
        
        # 6. σ^{-1}编码为多项式
        poly = self.sigma_inv(v_rounded)
        
        return poly
    
    def decode(self, poly: np.polynomial.Polynomial, scale: float) -> np.ndarray:
        """
        解码过程：将多项式解码为复向量
        
        解码流程：
        z = π ∘ σ(Δ^{-1}·m)
        
        Args:
            poly: 待解码的多项式
            scale: 缩放因子Δ
            
        Returns:
            解码后的复向量，长度为N/2
        """
        # 1. 缩放回原始尺度
        poly_scaled = poly / scale
        
        # 2. σ映射为向量
        vector = self.sigma(poly_scaled)
        
        # 3. π操作提取前N/2个元素
        z = self.pi(vector)
        
        return z
    
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
    
