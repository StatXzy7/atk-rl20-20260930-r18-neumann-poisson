# atk-rl20-20260930-r18-neumann-poisson

二维矩形均匀网格上 −div(k∇u)=f 的单元中心有限体积求解器，四边零法向
通量（纯 Neumann）。内部面系数取调和平均，算子由面通量装配（矩阵自
由），常数零空间用零均值约束处理，线性求解为手写的投影共轭梯度
（投影对角预条件）。

## 文件

- `fv_neumann.py` — 网格、算子（矩阵自由 `apply` + 独立稀疏装配
  `to_sparse`）、兼容性检查与投影 PCG 求解器
- `manufactured.py` — 满足零通量边界的光滑制造解及其源项单元平均
- `test_fv_neumann.py` — 20 项 pytest 测试（结构、兼容性、制造解、
  收敛阶、高对比、非法输入）
- `experiments.py` — 可复算实验脚本；输出快照在 `experiments_output.txt`
- `DERIVATION.md` — 离散格式、兼容条件、零空间与投影 CG 的完整推导

## 运行

```bash
python3 -m pytest test_fv_neumann.py -q
python3 experiments.py
```
