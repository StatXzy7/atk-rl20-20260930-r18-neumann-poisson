# 纯 Neumann 扩散问题的单元中心有限体积离散

## 1. 连续问题

在矩形域 \(\Omega=(0,L_x)\times(0,L_y)\) 上求

\[
-\nabla\cdot(k\nabla u)=f,\qquad
k\nabla u\cdot n=0\quad\text{on }\partial\Omega .
\]

单元系数 \(k_C>0\) 按单元给定。四边都是零法向通量，不设置任何 Dirichlet 值。

## 2. 面通量与离散方程

均匀网格步长为 \(\Delta x,\Delta y\)，单元体积 \(V=\Delta x\Delta y\)。x 方向内部面的面积为 \(\Delta y\)，y 方向内部面的面积为 \(\Delta x\)。

相邻单元 \(L,R\) 的扩散系数采用调和平均：

\[
k_f=\frac{2k_Lk_R}{k_L+k_R}.
\]

物理通量为 \(q=-k\nabla u\)。以 \(L\) 到 \(R\) 为面正向，左单元外法向为正、右单元外法向为负，因此

\[
q_f\cdot n_{L\to R}
=-t_f(u_R-u_L)=t_f(u_L-u_R),\qquad
t_f=k_f\frac{|f|}{d_L+d_R}.
\]

对单元 \(C\) 积分并除以 \(V\)，得到单位体积离散方程

\[
(Au)_C=\frac{1}{V}\sum_{f\subset C} q_f\cdot n_C=b_C.
\]

每条内部面在矩阵自由装配中同时加入两个相反贡献：左单元加入
\(t_f(u_L-u_R)/V\)，右单元加入 \(-t_f(u_L-u_R)/V\)。

因此所有内部面贡献在全局求和中两两抵消。边界通量按边界条件显式置零。

矩阵元为

\[
A_{CC}=\sum_{f\subset C}\frac{t_f}{V},\qquad
A_{CL}=-\frac{t_f}{V},\qquad A=A^\top.
\]

\(A\) 在零边界通量下半正定，且对任意常数向量 \(c\mathbf 1\) 有

\[
A(c\mathbf 1)=0.
\]

连通网格上常数函数是一维零空间：

\[
\ker(A)=\operatorname{span}\{\mathbf 1\}.
\]

## 3. 离散兼容条件与零均值代表解

对离散方程按单元体积求和：

\[
\mathbf 1^\top V A u=\mathbf 1^\top V b.
\]

由于面通量两两抵消且边界通量为零，左边恒为零。因此右端必须满足

\[
\sum_C V_C b_C=0,
\]

均匀网格上等价于普通算术平均 \(\overline b=0\)。

若解存在，\(u\) 和 \(u+c\) 都满足方程。采用约束

\[
\sum_C V_C u_C=0
\]

选择唯一代表。默认情况下，不满足兼容条件的右端直接抛出
`IncompatibleRightHandSide`。

只有显式选择 `rhs_mode="project"` 时才求解修改后的方程

\[
Au=b-\overline b\mathbf 1,
\]

并报告移除的均值 \(\overline b\)。程序不会静默改写右端。

## 4. 投影 PCG

定义体积权投影

\[
Pz=z-\frac{\mathbf 1^\top Vz}{\mathbf 1^\top V\mathbf 1}\mathbf 1.
\]

所有迭代量都保持在零均值约束子空间中：

- 初始值投影为 \(x_0=P x_0\)；
- 残差投影为 \(r=P(b-Ax)\)；
- Jacobi 预条件后的向量投影为 \(z=P(D^{-1}r)\)；
- 每次搜索方向递推后投影为 \(p=P(z+\beta p)\)。

核心递推为

\[
\alpha=\frac{r^\top z}{p^\top Ap},\quad
x\leftarrow x+\alpha p,\quad
r\leftarrow r-\alpha Ap,
\]

然后重新计算 \(z,\beta,p\)。该投影阻止舍入误差逐步漂入常数零空间。收敛后重新计算真实残差 \(\|b-Ax\|_2\)，不只使用求解器标志。

1×1 网格没有内部面，离散方程是 \(0=b\)；兼容时零均值代表解为 \(u=0\)。

## 5. 制造解和核验

离散制造解取任意零均值网格函数 \(u^\star\)，用同一离散算子生成

\[
b=Au^\star,
\]

再由投影 PCG 恢复零均值代表解。矩阵自由路径另外用独立稀疏装配矩阵和稠密 KKT 约束系统核验：

\[
\begin{bmatrix}
A & -V\\
V^\top & 0
\end{bmatrix}
\begin{bmatrix}u\\\lambda\end{bmatrix}
=
\begin{bmatrix}b\\0\end{bmatrix}.
\]

连续制造解在矩形上取

\[
u(x,y)=\cos(2\pi x/L_x)\cos(2\pi y/L_y),
\]

\[
k(x,y)=1+0.5\sin(\pi x/L_x)\cos(\pi y/L_y),
\]

二者均满足四边零法向导数。源项由 \(f=-\nabla\cdot(k\nabla u)\) 解析给出，并在每个单元内用 5×5 Gauss-Legendre 积分得到单元平均右端。积分右端本身满足离散零和；实验仍显式调用投影接口并报告移除均值，通常为精确零。

主要文件：

- `neumann_fv/grid.py`：均匀矩形网格。
- `neumann_fv/operator.py`：调和平均、面通量、矩阵自由装配和独立稀疏矩阵。
- `neumann_fv/solver.py`：兼容性检查、投影和自实现 PCG。
- `neumann_fv/mms.py`：连续制造解、源积分和独立 KKT 求解。
- `run_experiments.py`：可复现实验，输出 `results.json`。
