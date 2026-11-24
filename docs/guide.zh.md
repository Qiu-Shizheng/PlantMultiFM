# PlantMultiFM 使用指南

安装软件并从[发布页](https://github.com/Qiu-Shizheng/PlantMultiFM/releases/tag/v0.1.0)下载与物种、参考基因组和实验特征轴匹配的模型包。三个包使用相同的预训练权重。下载后根据 SHA256SUMS 核验文件。首次运行示例见 [README](../README.md)。

输入矩阵为“细胞 × 特征”。RNA 使用原始非负整数计数，ATAC 使用二值可及性。矩阵列名必须唯一，顺序与特征文件逐一对应。保留已测量但全零的特征，剔除未测量或无法映射的列。不要将未测量的基因或区域补零。峰坐标须来自同一参考组装。

- `encode`：生成每个细胞的 256 维表示，可在带标注的参考库上训练分类器，再给新细胞注释。
- `atac_to_rna`：从 ATAC 返回原生表达分数。评估某一转录本在哪些细胞中检出时，使用该分数在该基因内对细胞排序。
- `atac_to_rna_detection`：返回独立检出头的 sigmoid 分数。
- `rna_to_atac`：从 RNA 为指定区域输出可及性分数。
- `masked_rna_detection`：从其余可见基因预测目标基因检出。先删除目标基因列，再计算细胞表示。

`plantmultifm features` 列出参考轴。`embed` 导出细胞表示，`predict` 导出细胞与目标的分数矩阵。输出行顺序与输入相同，输出列顺序与目标列表相同。[完整示例及 AnnData 接口](guide.md)。

