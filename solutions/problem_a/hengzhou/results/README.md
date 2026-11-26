# 横州第 3、4、5 问的实验结果

| 问题 | 主要结果 | 查看文件 |
| --- | --- | --- |
| 第 3 问 · 水体变化 | 7 月 4 日至 6 日，共同可判读的 57.781 km² 中新增淹没 19.251 km²，净增 19.245 km²。 | [淹没图](Q3_event_map.png) · [面积与敏感性分析](q3_results.json) · [12 个分区明细](q3_units.csv) · [新增淹没边界](gain_polygons.geojson) |
| 第 3 问 · 模型迁移 | 在 7 月 7 日快视图的 55.31 km² 有效范围，直接使用显示灰度预测新增 0.130 km²，直方图匹配后预测 0.046 km²。两种预测在同日解译对照区的空间交并比均为 0。 | [横州实景测试指标](q3_model_transfer.json) · [直接灰度预测](July07_Q2_display_dn.tif) · [直方图匹配预测](July07_Q2_train_histogram.tif) |
| 第 4 问 · 灾情评估 | 估计 4,567 名居民位于可见新增区，9.69 km 已制图道路与其相交。四组权重下，r5c6、r5c5 均居前两位。 | [风险排名表](q4_risk.csv) · [优先级地图](Q4_priority_map.png) · [带边界的评分数据](q4_risk.geojson) |
| 第 5 问 · 物资分配 | 在首日库存 3,000 份、每区最低覆盖 45% 的假设下，分区总需求 4,575 份，剩余缺口 1,575 份。 | [逐区分配明细](q5_allocation.csv) · [与按需比例分配的比较及九组情景](q45_results.json) |

## 第 3 问使用的模型

第 3 问复用第 2 问在本地训练的融合网络。训练共 24 轮，选中第 17 轮，预测阈值为 0.65。原训练数据的独立测试集含 203 个切片，面积加权交并比为 0.6591。横州快视图上的实测指标单独记录在上表。第 4 问使用空间叠加和加权评分，第 5 问使用整数规划。

[模型权重](q3_frozen_model/fusion_best.pt) · [选中轮次和阈值](q3_frozen_model/frozen_model.json) · [逐轮训练曲线](q3_frozen_model/training_curve.csv) · [原数据集评估指标](q3_frozen_model/fusion_metrics.json) · [权重复核记录](q3_frozen_model/verification.json)

## 图层与来源

7 月 6 日面积统计来自公开分类图与灾前水体的叠加。变化栅格中 0 表示两期均干，1 表示新增水体，2 表示消退，3 表示持续水体，255 表示未知。[灾前候选水体](July04_radar_water_candidate.tif) · [灾中可见水体](July06_visible_water.tif) · [两期变化栅格](July06_transitions.tif)

灾中图层由 Shan 等（2026）的图 3 经配准、裁切和分类色提取得到，原图按 CC BY 4.0 发布。7 月 7 日的快视图由电建一号研究团队提供。[原始论文](https://doi.org/10.3390/rs18234011) · [快视图报道](https://news.sciencenet.cn/htmlnews/2026/7/567982.shtm)

人口估计使用 WorldPop Global2 R2025A（CC BY 4.0），道路来自 © OpenStreetMap contributors（ODbL）的广西快照。运行步骤及灾前卫星数据来源见[程序说明](../README.md)。
