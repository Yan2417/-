# A 题第一、二问：新增淹没区识别与强度分级

横州实景制图、灾情评分与救援物资分配见[第 3 至 5 问程序与运行说明](hengzhou/README.md)。

第一问以随机森林识别新增淹没像素。第二问用双分支 U-Net 融合灾前光学与灾后 SAR 影像，比较面积误差与空间重叠，并把面积不确定性传递到四级淹没强度。标签 1 按新增淹没区处理。题面未列出类别值，判读依据是先看灾前影像选取河湖，再核查了 9 组灾前已有水体的样点和邻近受淹地块。

## 论文用实验结果

[下载第一、二问完整实验交付包](artifacts/Q1-Q2-experiments.zip)

两种模型在同一批 203 张独立测试切片上评估。第一问汇总指标见 `q1/metrics.json`，第二问见 `q2/fusion_metrics.json` 的 `test` 部分。逐片对比可按 `id` 合并两问的 `test_tiles.csv`。分地表误差见 `q2/assessment/stratified_errors.csv`，逐片强度等级与不确定区间见 `q2/assessment/intensity_by_tile.csv`。包内还含训练代码、冻结权重、划分清单和逐像素预测。

## 复现已运行的实验

运行环境为 Linux、Python 3.11、PyTorch 2.1.0（实验使用 CUDA 12.1）。其余依赖版本见 requirements.txt。先将竞赛附件解压。数据根目录下应有 `pre-disaster optical images`、`post-disaster SAR images`、`label information` 三个子目录，每个目录含编号 1001–2000 的 1,000 张 TIFF。原始附件 1 存放在实验室服务器，训练好的模型权重收录在实验交付包中。

从本代码目录执行，在自己的 Python 环境中安装依赖。把 DATA 改为附件 1 的绝对路径，把 RUN 改为一个新的实验目录。

```bash
CODE="$(pwd)"
DATA="/path/to/Attachment 1 Model Training Dataset"
RUN="/path/to/new-experiment"
python -m pip install -r "$CODE/requirements.txt"
mkdir -p "$RUN/q1" "$RUN/q2"
cd "$RUN/q1"
python "$CODE/baseline.py" --source "$DATA"
python "$CODE/verify.py"
cd "$RUN/q2"
python "$CODE/fusion.py" --baseline "$RUN/q1"
python "$CODE/verify_fusion.py" --baseline "$RUN/q1" --fusion "$RUN/q2" --out "$RUN/q2/verification.json"
python "$CODE/worldcover.py" --baseline "$RUN/q1" --out "$RUN/worldcover"
python "$CODE/assessment.py" --baseline "$RUN/q1" --fusion "$RUN/q2" --wc "$RUN/worldcover" --out "$RUN/assessment"
```

使用 GPU 复现实验时保留默认推理批量 12。不同 GPU 或底层库可能带来浮点舍入差异。地表分层步骤需要联网，按字节范围下载测试区覆盖的 31 个图块。ESA WorldCover 2021 v200 来源与压缩数据 SHA-256 会写入来源清单。该图用于事后误差归因，均未作为两种识别模型的训练输入。其年份与事件影像的获取日期需区别解释。[ESA 产品与许可说明](https://esa-worldcover.org/en/data-access)

## 实验设计与结果

- 配准使用 WGS84 原始标签网格。对灾前 RGB 和灾后 SAR 双线性取值，保留原始标签。有效掩膜剔除影像空白、越界和标签无效像素。按 WGS84 椭球每一行的实际像元面积积分。
- 划分为训练 602 张、验证 137 张、独立测试 203 张，另有空间缓冲 58 张。训练和测试影像边界至少相隔 2.02 km。第一问的阈值由验证集选取。第二问将原验证集按空间进一步分为选择集 64 张、误差校准集 59 张和缓冲 14 张。沿用相同的 203 张测试切片以便配对比较。
- 随机森林使用 12 项特征、48 棵树及 306,673 个训练像素，判为正类的阈值为 26 票。融合网络用分支卷积分别编码 RGB 与原始/中值/局部标准差 SAR，再利用编码器、跳跃连接和解码器识别多尺度结构。用掩膜 BCE、Dice 损失与逐片面积比例误差训练；以选择集 WAPE + 1 − IoU 最小为准，24 轮中选第 8 轮，阈值 0.50。
- 测试集标签新增面积 190.98 km²。随机森林预测 155.80 km²，融合网络预测 167.74 km²。面积加权 IoU 从 0.6484 变为 0.7374，WAPE 从 26.85% 变为 20.10%。西部留出区 IoU 从 0.2388 降至 0.2161，小于 1 公顷的斑块召回率从 29.3% 降至 21.3%。
- 四级按 436 张完整且有新增淹没的训练切片的占比分布标定。取中位数、第 75 和第 90 百分位，三个经验界点约为 7.97%、46.17%、98.18%。边界取上一级，计算使用 JSON 中的未舍入值。对面积为 S 的研究区，面积阈值分别等于这些比例乘以 S。校准区目标覆盖率为 90%，独立测试的有效区比例区间实测覆盖率为 77.83%；这些经验分级适用于与训练切片尺度相近的区域。
- Q1 的 `registered` 保存配准影像与掩膜，各自实验的 `test_predictions` 保存二值预测（255 为无效）。`test_tiles.csv` 可逐片核查。`assessment` 包含按地表和雷达亮度分层的误报/漏报、连通斑块召回率、难例子集表现、逐片等级、面积上下界和等级混淆矩阵。整体与分区指标也记录在 results_snapshot.json。
