# BAF-SynPred 最终可复现项目

本目录只保留论文最终采用、保留 CrossTalk 的主模型，以及直接复现所需的
源码、10,183 条样本数据、977 维细胞系特征、固定五折划分和五个最佳权重。
历史版本、消融实验、超参数实验、多组学试验、绘图代码和论文文件均未纳入。

## 两步运行

在本目录打开 PowerShell：

```powershell
.\install_windows.ps1
.\run_evaluation.ps1
```

首次评价会自动核验文件哈希、检查数据并生成图缓存，然后加载五个权重完成
五折评价；结果保存为 `runs/canonical_5fold_evaluation.json`。

如需先做快速检查：

```powershell
.\run_preflight.ps1
```

如需重新训练完整五折：

```powershell
.\run_training.ps1 -HighPriority
```

最终默认参数为：`embed_dim=128`、`bond_k=3`、细胞噪声 `0.05`、
cell-drug dropout `0.2`、adaptive KL beta `0.5`、batch size `64`、
学习率 `2e-4`、weight decay `0`、重构损失权重 `0.01`。

随附权重对应 `20260908_225152_bafsynpred`，五折 ROC-AUC 为
`0.981064 +/- 0.003102`。该结果是固定划分下的五折 OOF 结果，不是独立测试集。

公开上传前，请确认原始数据的再分发许可，并由作者自行确定软件许可证；本次
整理不擅自指定许可证。

