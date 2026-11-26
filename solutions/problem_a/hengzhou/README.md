# 横州洪灾建模 · A 题第 3 至 5 问

第 3 问分别计算 7 月 6 日的水体变化、检验第 2 问网络在 7 月 7 日雷达快视图上的表现。第 4 问使用 6 日的新增淹没多边形估算人口和道路暴露。第 5 问按分区风险和需求分配首日物资。

| 程序 | 用途 | 主要输出 |
| --- | --- | --- |
| `event_map.py` | 将灾前水体与 6 日公开卫星分类图对齐，统计新增、消退和净增面积 | `July06_transitions.tif`、`gain_polygons.geojson`、`q3_results.json` |
| `transfer.py` | 在 7 日快视图上运行冻结的第 2 问网络，并比较两种灰度处理结果 | `q3_model_transfer.json`、两张预测栅格 |
| `predict_scene.py` | 为已配准的三通道灾前影像和训练量纲的单通道 HH 雷达影像提供推理入口 | 地理配准的二值预测栅格，255 表示未知 |
| `people.py` | 按新增淹没多边形汇总 WorldPop 人口，同一边界复用已保存的查询 | `worldpop_queries/*.json` |
| `assessment_allocation.py` | 计算 12 个单元的灾情评分、整数物资分配及库存和需求敏感性 | `q4_risk.csv`、`q5_allocation.csv`、`q45_results.json` |
| `plot_maps.py` | 绘制水体变化图和分区优先级图 | `Q3_event_map.png`、`Q4_priority_map.png` |

## 运行

使用 Python 3.11。先解压团队的 `Q345_建模成果.zip`。保留解压后 `A题实验` 下的 `Q345`、`hengzhou`、`q2`、`assessment` 和 `exposure` 五个同级目录。影像、冻结权重、人口查询及道路底图在成果包中。程序从**当前工作目录**取数，并将输出写回该目录，复算时可使用一份解压副本。

在 PowerShell 中，从本仓库根目录执行。把第三行的路径改为本机解压位置。

```powershell
$programs = (Resolve-Path "solutions/problem_a/hengzhou").Path
python -m pip install -r "$programs/requirements.txt"
Set-Location "D:/解压位置/A题实验/Q345"
python "$programs/event_map.py"
python "$programs/transfer.py"
python "$programs/people.py"
python "$programs/assessment_allocation.py"
python "$programs/plot_maps.py"
```

完整成果包已保存快视图配准结果及 12 个单元的人口查询响应。使用原研究区可离线复算。若改变水体边界，人口程序会向 WorldPop 重新查询。若需重做快视图配准，在 `sources` 下放入原始 `operator_0707.png` 和与 `reference_extent.json` 对应的 `registration_reference.jpg`，运行 `transfer.py --refit-registration`。

## 结果核对

共同可判读区新增淹没面积为 19.251 km²，来自 6 日公开分类图与灾前水体的比较。第 2 问网络在 7 日快视图上的两种预测分别为 0.130 和 0.046 km²。该快视图的地面采样约 96.5 m，与训练数据约 10 m 的尺度差异较大。第 4 问估计约 4,567 名居民位于可见新增区。第 5 问假设库存为 3,000 份，每区最低覆盖率为 45%，分区合计应分配 3,000 份，剩余需求为 1,575 份。

6 日和 7 日的公开水体解译见 [Shan 等的研究与图 3、图 5](https://www.mdpi.com/2072-4292/18/23/4011)。灾前影像来自 [Copernicus Sentinel 产品目录](https://planetarycomputer.microsoft.com/catalog)。7 日快视图由[电建一号研究团队提供](https://news.sciencenet.cn/htmlnews/2026/7/567982.shtm)，配准底图来自 [Esri World Imagery](https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer)。人口采用 [WorldPop Global2 R2025A](https://api.worldpop.org/v3/documentation)，道路采用 [Geofabrik 广西快照](https://download.geofabrik.de/asia/china/guangxi/)（© OpenStreetMap contributors，ODbL）。
