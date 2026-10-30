# 《第三讲 最近邻与决策树》来源记录

- 课程：模式识别与机器学习；主讲：刘乐茂；机构：复旦大学计算与智能创新学院。
- 原件：[3 最近邻与决策树.pdf](</Users/gary/大学/大三上/模式识别与机器学习/3 最近邻与决策树.pdf>)；位于 Wiki 外部，未修改。
- 页数：51；版本/发布日期：未知；读取与整理日期：2026-10-28。
- 读取范围：全51页通过 PDFKit 渲染并使用 macOS Vision OCR；原PDF没有可提取文本。
- 保存材料：[完整OCR快照](nearest-neighbors-decision-trees.ocr.txt)。保留原始识别结果，数学符号、表格顺序可能有误，不能直接当作准确公式。
- 视觉核对范围：额外查看 P15、20、37、38、39、42 的页面图，核对评价公式、余弦公式、熵、增益、增益比及不纯度对比。未逐页核对所有图表，不推断未读图表细节。
- 内容定位：P5–15分类与评价；P17–25 KNN；P27–29规则方法；P31–42决策树生成；P43–46剪枝与优缺点；P47–49 Xbox人体部位分类应用；P51总结。

## 课件与补充的边界

课件提供算法概念和切分准则。本次对话的网格数量、超立方体体积例子、6橙4柠的切分计算、信息增益等于互信息的解释由 Codex 补充，是教学示例，不是用户实验或课件原始数据。

已核对的课件问题：P15准确率分母漏项，且将 Precision 的 0/0 直接写成1；P20标题写余弦距离，公式实际为余弦相似度；P39“惩罚子节点数”仅为增益比的简化说法，分母实际依赖分支样本比例。修正见[第三讲导航](../wiki/nearest-neighbors-decision-trees.md)与[决策树专题](../wiki/decision-trees.md)。

## 补充来源：scikit-learn 官方文档

- 作者/机构：scikit-learn developers。
- [DecisionTreeClassifier API](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)：读取参数 `criterion`、`max_depth`、`min_samples_leaf` 等相关说明；默认 `gini`，`entropy` 与 `log_loss` 使用香农信息增益。
- [Decision Trees 用户指南](https://scikit-learn.org/stable/modules/tree.html)：读取分类不纯度、回归准则、贪心切分与剪枝相关部分。
- 核对日期：2026-10-28；网页显示版本：1.9.1；仅保存链接，未保存正文。未在本地安装或运行该版本。
- 这些证据可说明实现选择，不能据此断言全部实践或教学的流行趋势。

## 处理记录

2026-10-28：首次收录来源与OCR，建立[第三讲导航及KNN](../wiki/nearest-neighbors-decision-trees.md)、[决策树专题](../wiki/decision-trees.md)，并保存[学习过程](../journal/2026-10-28-最近邻与决策树.md)。
