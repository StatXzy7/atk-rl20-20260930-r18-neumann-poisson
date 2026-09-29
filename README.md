# atk-rl20-20260930-r18-neumann-poisson

二维矩形均匀网格上的纯 Neumann 扩散问题：

\[
-\nabla\cdot(k\nabla u)=f,\qquad k\nabla u\cdot n=0.
\]

实现使用单元中心有限体积法：内部面扩散系数为相邻单元系数的调和平均，矩阵自由算子由面通量散度装配，内部面贡献两两抵消，边界通量显式为零。常数零空间通过零均值约束选择代表解，不使用 Dirichlet 条件。

## 快速运行

```bash
python3 -m pytest -q
python3 run_experiments.py > results.json
```

## 文件

- `DERIVATION.md`：有限体积、兼容条件、零空间和投影 PCG 推导。
- `RESULTS.md`：复算实验、收敛阶、真残差、均值和守恒误差。
- `neumann_fv/`：求解器实现。
- `tests/`：离散制造解、独立稀疏/KKT 核验、异常输入和收敛测试。

默认拒绝不兼容右端；只有显式传入 `rhs_mode="project"` 才求解均值投影后的修改方程，并报告移除的均值。
