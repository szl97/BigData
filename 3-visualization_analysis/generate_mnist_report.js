const fs = require('fs');
const { Document, Packer, Paragraph, TextRun, ImageRun, AlignmentType, HeadingLevel, 
        BorderStyle, WidthType, ShadingType, VerticalAlign, PageBreak, LevelFormat, 
        TableOfContents, Table, TableRow, TableCell } = require('docx');

console.log("="*80);
console.log("生成MNIST手写数字识别数据集分析报告");
console.log("="*80);

// 读取所有可视化图片
console.log("\n[1/2] 读取可视化图表...");
const images = {};
const imageFiles = [
    '01_digit_samples.png',
    '02_pixel_distribution.png',
    '03_pixel_importance.png',
    '04_pca_analysis.png',
    '05_tsne_visualization.png',
    '06_digit_similarity.png',
    '07_feature_importance.png',
    '08_confusion_matrix.png',
    '09_clustering_analysis.png'
];

imageFiles.forEach(file => {
    const filepath = `./mnist_visualizations/${file}`;
    if (fs.existsSync(filepath)) {
        images[file] = fs.readFileSync(filepath);
        console.log(`  ✓ ${file}`);
    } else {
        console.log(`  ✗ 未找到: ${file}`);
    }
});

console.log("\n[2/2] 生成Word文档...");

// 创建文档
const doc = new Document({
    styles: {
        default: { document: { run: { font: "Arial", size: 24 } } },
        paragraphStyles: [
            { id: "Title", name: "Title", basedOn: "Normal",
              run: { size: 56, bold: true, color: "2C3E50", font: "Arial" },
              paragraph: { spacing: { before: 240, after: 240 }, alignment: AlignmentType.CENTER } },
            { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
              run: { size: 36, bold: true, color: "2C3E50", font: "Arial" },
              paragraph: { spacing: { before: 360, after: 240 }, outlineLevel: 0 } },
            { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
              run: { size: 30, bold: true, color: "34495E", font: "Arial" },
              paragraph: { spacing: { before: 300, after: 180 }, outlineLevel: 1 } },
            { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true,
              run: { size: 26, bold: true, color: "7F8C8D", font: "Arial" },
              paragraph: { spacing: { before: 240, after: 120 }, outlineLevel: 2 } }
        ]
    },
    numbering: {
        config: [
            { reference: "bullet-list", levels: [
                { level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
                  style: { paragraph: { indent: { left: 720, hanging: 360 } } } }
            ]},
            { reference: "numbered-list", levels: [
                { level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT,
                  style: { paragraph: { indent: { left: 720, hanging: 360 } } } }
            ]}
        ]
    },
    sections: [{
        properties: {
            page: { margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 } }
        },
        children: [
            // 封面
            new Paragraph({ heading: HeadingLevel.TITLE, children: [
                new TextRun("大数据处理与可视化分析")
            ]}),
            new Paragraph({ heading: HeadingLevel.TITLE, children: [
                new TextRun("作业三：高维数据可视化分析报告")
            ]}),
            new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 600, after: 200 },
                children: [new TextRun({ text: "MNIST手写数字识别数据集分析", size: 32, bold: true, color: "E74C3C" })] }),
            new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 100, after: 200 },
                children: [new TextRun({ text: "基于Kaggle公开数据集", size: 24, italics: true })] }),
            new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 200, after: 800 },
                children: [new TextRun({ text: "2025年11月", size: 24 })] }),

            // 目录
            new Paragraph({ children: [new PageBreak()] }),
            new TableOfContents("目录", { hyperlink: true, headingStyleRange: "1-3" }),
            new Paragraph({ children: [new PageBreak()] }),

            // 第一章：引言
            new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun("1. 引言")] }),
            new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun("1.1 研究背景")] }),
            new Paragraph({ spacing: { after: 120 }, children: [
                new TextRun("手写数字识别是计算机视觉领域的经典问题，在邮政编码识别、银行支票处理、表单数字化等场景有广泛应用。MNIST（Modified National Institute of Standards and Technology）数据集是手写数字识别领域最著名的基准数据集之一，包含70,000张28×28像素的手写数字图像，每张图像对应0-9中的一个数字。")
            ]}),
            new Paragraph({ spacing: { after: 120 }, children: [
                new TextRun("本报告通过对MNIST数据集进行全面的可视化分析，深入探讨手写数字的特征分布、类别可分性、降维效果以及分类难点，为理解图像识别任务和机器学习算法提供数据洞察。")
            ]}),

            new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun("1.2 数据集来源")] }),
            new Paragraph({ spacing: { after: 120 }, children: [
                new TextRun({ text: "数据来源：", bold: true }),
                new TextRun("Kaggle公开数据集（https://www.kaggle.com/datasets/oddrationale/mnist-in-csv）")
            ]}),
            new Paragraph({ numbering: { reference: "bullet-list", level: 0 }, children: [
                new TextRun("训练集：60,000个样本")
            ]}),
            new Paragraph({ numbering: { reference: "bullet-list", level: 0 }, children: [
                new TextRun("测试集：10,000个样本")
            ]}),
            new Paragraph({ numbering: { reference: "bullet-list", level: 0 }, children: [
                new TextRun("总样本数：70,000")
            ]}),
            new Paragraph({ numbering: { reference: "bullet-list", level: 0 }, children: [
                new TextRun("特征维度：784（28×28像素）")
            ]}),

            // 第二章：问题定义
            new Paragraph({ children: [new PageBreak()] }),
            new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun("2. 问题定义与研究目标")] }),
            new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun("2.1 核心问题")] }),
            new Paragraph({ spacing: { after: 60 }, children: [
                new TextRun("本研究聚焦于手写数字识别任务的数据特征分析，主要回答以下问题：")
            ]}),
            new Paragraph({ numbering: { reference: "numbered-list", level: 0 }, children: [
                new TextRun({ text: "特征分布：", bold: true }),
                new TextRun("手写数字在像素空间中的分布特征是什么？哪些像素位置最重要？")
            ]}),
            new Paragraph({ numbering: { reference: "numbered-list", level: 0 }, children: [
                new TextRun({ text: "类别可分性：", bold: true }),
                new TextRun("不同数字之间的相似度如何？哪些数字容易混淆？")
            ]}),
            new Paragraph({ numbering: { reference: "numbered-list", level: 0 }, children: [
                new TextRun({ text: "降维效果：", bold: true }),
                new TextRun("能否用更少的特征表示手写数字？降维后能保留多少信息？")
            ]}),
            new Paragraph({ numbering: { reference: "numbered-list", level: 0 }, children: [
                new TextRun({ text: "分类难点：", bold: true }),
                new TextRun("机器学习模型在识别手写数字时面临哪些挑战？")
            ]}),

            new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun("2.2 研究意义")] }),
            new Paragraph({ spacing: { after: 120 }, children: [
                new TextRun("通过可视化分析MNIST数据集，我们可以：")
            ]}),
            new Paragraph({ numbering: { reference: "bullet-list", level: 0 }, children: [
                new TextRun("理解图像数据的本质特征和分布规律")
            ]}),
            new Paragraph({ numbering: { reference: "bullet-list", level: 0 }, children: [
                new TextRun("为模型选择和特征工程提供数据依据")
            ]}),
            new Paragraph({ numbering: { reference: "bullet-list", level: 0 }, children: [
                new TextRun("识别数据中的潜在问题和挑战")
            ]}),
            new Paragraph({ numbering: { reference: "bullet-list", level: 0 }, children: [
                new TextRun("为计算机视觉任务提供可视化分析范例")
            ]}),

            // 第三章：数据处理
            new Paragraph({ children: [new PageBreak()] }),
            new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun("3. 数据处理过程")] }),
            new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun("3.1 数据格式")] }),
            new Paragraph({ spacing: { after: 120 }, children: [
                new TextRun("MNIST数据集为CSV格式，每行代表一个样本：")
            ]}),
            new Paragraph({ numbering: { reference: "bullet-list", level: 0 }, children: [
                new TextRun({ text: "第1列：", bold: true }),
                new TextRun("label（数字标签，0-9）")
            ]}),
            new Paragraph({ numbering: { reference: "bullet-list", level: 0 }, children: [
                new TextRun({ text: "第2-785列：", bold: true }),
                new TextRun("784个像素值（28×28图像展平，取值0-255）")
            ]}),

            new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun("3.2 预处理步骤")] }),
            new Paragraph({ numbering: { reference: "numbered-list", level: 0 }, children: [
                new TextRun({ text: "像素归一化：", bold: true }),
                new TextRun("将像素值从[0, 255]缩放到[0, 1]，便于模型训练和可视化")
            ]}),
            new Paragraph({ numbering: { reference: "numbered-list", level: 0 }, children: [
                new TextRun({ text: "数据标准化：", bold: true }),
                new TextRun("Z-score标准化，用于PCA降维和聚类分析")
            ]}),
            new Paragraph({ numbering: { reference: "numbered-list", level: 0 }, children: [
                new TextRun({ text: "采样策略：", bold: true }),
                new TextRun("为提高计算效率，使用10,000个训练样本进行分析（完整数据集70,000个）")
            ]}),

            new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun("3.3 特征提取")] }),
            new Paragraph({ spacing: { after: 120 }, children: [
                new TextRun("从原始像素数据中提取了多种特征用于分析：")
            ]}),
            new Paragraph({ numbering: { reference: "bullet-list", level: 0 }, children: [
                new TextRun("像素统计特征：均值、方差、最大值、最小值")
            ]}),
            new Paragraph({ numbering: { reference: "bullet-list", level: 0 }, children: [
                new TextRun("PCA主成分：提取关键的线性组合特征")
            ]}),
            new Paragraph({ numbering: { reference: "bullet-list", level: 0 }, children: [
                new TextRun("t-SNE嵌入：捕捉数据的非线性流形结构")
            ]}),
            new Paragraph({ numbering: { reference: "bullet-list", level: 0 }, children: [
                new TextRun("随机森林特征重要性：识别对分类贡献最大的像素")
            ]}),

            // 第四章：可视化分析
            new Paragraph({ children: [new PageBreak()] }),
            new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun("4. 可视化分析与呈现")] }),

            // 4.1 手写数字样本展示
            new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun("4.1 手写数字样本展示")] }),
            new Paragraph({ spacing: { after: 240 }, children: [
                new TextRun("首先展示MNIST数据集中的真实手写数字样本。图4.1展示了0-9每个数字的10个随机样本，可以直观看到：（1）同一数字的不同书写风格和变化；（2）数字间的视觉差异；（3）数据的质量和多样性。")
            ]}),
            new Paragraph({ alignment: AlignmentType.CENTER, children: [
                new ImageRun({
                    type: "png", data: images['01_digit_samples.png'],
                    transformation: { width: 600, height: 600 },
                    altText: { title: "手写数字样本", description: "MNIST数字样本展示", name: "samples" }
                })
            ]}),
            new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 120, after: 360 }, children: [
                new TextRun({ text: "图4.1 MNIST手写数字样本（每个数字10个样本）", italics: true, size: 20 })
            ]}),
            new Paragraph({ spacing: { after: 120 }, children: [
                new TextRun({ text: "关键观察：", bold: true })
            ]}),
            new Paragraph({ numbering: { reference: "bullet-list", level: 0 }, children: [
                new TextRun("数字1和7结构简单，样本间差异小")
            ]}),
            new Paragraph({ numbering: { reference: "bullet-list", level: 0 }, children: [
                new TextRun("数字4、5、9书写风格多样，存在较大变异")
            ]}),
            new Paragraph({ numbering: { reference: "bullet-list", level: 0 }, children: [
                new TextRun("某些样本存在笔画粗细、倾斜角度、位置偏移等差异")
            ]}),

            // 4.2 像素分布分析
            new Paragraph({ children: [new PageBreak()] }),
            new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun("4.2 像素强度分布分析")] }),
            new Paragraph({ spacing: { after: 240 }, children: [
                new TextRun("图4.2分析了像素值的统计分布特征，左上角展示整体像素值分布，其余子图展示每个数字的平均图像（热力图）。通过平均图像可以看出每个数字的典型形状和关键笔画位置。")
            ]}),
            new Paragraph({ alignment: AlignmentType.CENTER, children: [
                new ImageRun({
                    type: "png", data: images['02_pixel_distribution.png'],
                    transformation: { width: 600, height: 333 },
                    altText: { title: "像素分布", description: "像素强度分布", name: "distribution" }
                })
            ]}),
            new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 120, after: 360 }, children: [
                new TextRun({ text: "图4.2 像素强度分布与平均数字图像", italics: true, size: 20 })
            ]}),
            new Paragraph({ spacing: { after: 120 }, children: [
                new TextRun({ text: "分析结果：", bold: true })
            ]}),
            new Paragraph({ numbering: { reference: "bullet-list", level: 0 }, children: [
                new TextRun("像素值呈现明显的双峰分布：大量0值（背景）和一些较高值（笔画）")
            ]}),
            new Paragraph({ numbering: { reference: "bullet-list", level: 0 }, children: [
                new TextRun("平均数字图像清晰展示各数字的典型形态")
            ]}),
            new Paragraph({ numbering: { reference: "bullet-list", level: 0 }, children: [
                new TextRun("数字1的平均图像最清晰（变化小），数字2、3、5较模糊（变化大）")
            ]}),

            // 4.3 像素重要性
            new Paragraph({ children: [new PageBreak()] }),
            new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun("4.3 像素重要性热力图")] }),
            new Paragraph({ spacing: { after: 240 }, children: [
                new TextRun("图4.3通过统计分析识别出最重要的像素区域。左图展示像素方差（方差越大，信息量越丰富），右图展示平均像素强度。")
            ]}),
            new Paragraph({ alignment: AlignmentType.CENTER, children: [
                new ImageRun({
                    type: "png", data: images['03_pixel_importance.png'],
                    transformation: { width: 550, height: 275 },
                    altText: { title: "像素重要性", description: "像素方差和强度", name: "importance" }
                })
            ]}),
            new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 120, after: 360 }, children: [
                new TextRun({ text: "图4.3 像素重要性分析（方差与强度）", italics: true, size: 20 })
            ]}),
            new Paragraph({ spacing: { after: 120 }, children: [
                new TextRun({ text: "核心发现：", bold: true })
            ]}),
            new Paragraph({ numbering: { reference: "bullet-list", level: 0 }, children: [
                new TextRun({ text: "边缘区域（4像素宽度）几乎不含信息，可以裁剪", bold: true })
            ]}),
            new Paragraph({ numbering: { reference: "bullet-list", level: 0 }, children: [
                new TextRun("中心区域（约20×20像素）包含大部分区分信息")
            ]}),
            new Paragraph({ numbering: { reference: "bullet-list", level: 0 }, children: [
                new TextRun("高方差区域对应数字笔画常出现的位置")
            ]}),

            // 4.4 PCA降维
            new Paragraph({ children: [new PageBreak()] }),
            new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun("4.4 PCA降维分析")] }),
            new Paragraph({ spacing: { after: 240 }, children: [
                new TextRun("主成分分析(PCA)将784维像素特征投影到2维空间。左图展示降维后的数据分布，右图分析需要多少主成分才能保留足够的信息。")
            ]}),
            new Paragraph({ alignment: AlignmentType.CENTER, children: [
                new ImageRun({
                    type: "png", data: images['04_pca_analysis.png'],
                    transformation: { width: 600, height: 225 },
                    altText: { title: "PCA分析", description: "主成分分析", name: "pca" }
                })
            ]}),
            new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 120, after: 360 }, children: [
                new TextRun({ text: "图4.4 PCA降维分析与方差解释", italics: true, size: 20 })
            ]}),
            new Paragraph({ spacing: { after: 120 }, children: [
                new TextRun({ text: "重要结果：", bold: true })
            ]}),
            new Paragraph({ numbering: { reference: "bullet-list", level: 0 }, children: [
                new TextRun("前2个主成分仅解释约10%的方差，说明数据高度非线性")
            ]}),
            new Paragraph({ numbering: { reference: "bullet-list", level: 0 }, children: [
                new TextRun({ text: "保留95%信息需要283个主成分（降维率64%）", bold: true })
            ]}),
            new Paragraph({ numbering: { reference: "bullet-list", level: 0 }, children: [
                new TextRun("保留99%信息需要489个主成分（降维率38%）")
            ]}),
            new Paragraph({ numbering: { reference: "bullet-list", level: 0 }, children: [
                new TextRun("PCA投影中数字0、1聚集较紧密，4、7、9较分散")
            ]}),

            // 4.5 t-SNE
            new Paragraph({ children: [new PageBreak()] }),
            new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun("4.5 t-SNE非线性降维")] }),
            new Paragraph({ spacing: { after: 240 }, children: [
                new TextRun("t-SNE是一种非线性降维技术，能更好地保留数据的局部结构。图4.5展示了3000个样本的t-SNE投影，不同颜色代表不同数字。")
            ]}),
            new Paragraph({ alignment: AlignmentType.CENTER, children: [
                new ImageRun({
                    type: "png", data: images['05_tsne_visualization.png'],
                    transformation: { width: 500, height: 417 },
                    altText: { title: "t-SNE", description: "t-SNE降维", name: "tsne" }
                })
            ]}),
            new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 120, after: 360 }, children: [
                new TextRun({ text: "图4.5 t-SNE可视化（3000个样本）", italics: true, size: 20 })
            ]}),
            new Paragraph({ spacing: { after: 120 }, children: [
                new TextRun({ text: "关键洞察：", bold: true })
            ]}),
            new Paragraph({ numbering: { reference: "bullet-list", level: 0 }, children: [
                new TextRun({ text: "数字1形成最紧密的簇，说明书写风格一致", bold: true })
            ]}),
            new Paragraph({ numbering: { reference: "bullet-list", level: 0 }, children: [
                new TextRun("数字0形成清晰的环形簇，与其他数字分离良好")
            ]}),
            new Paragraph({ numbering: { reference: "bullet-list", level: 0 }, children: [
                new TextRun("数字4、7、9分布较分散，类内变异大")
            ]}),
            new Paragraph({ numbering: { reference: "bullet-list", level: 0 }, children: [
                new TextRun("数字3和5、4和9有部分重叠区域，可能易混淆")
            ]}),

            // 4.6 相似度矩阵
            new Paragraph({ children: [new PageBreak()] }),
            new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun("4.6 数字间相似度分析")] }),
            new Paragraph({ spacing: { after: 240 }, children: [
                new TextRun("通过计算每个数字的平均图像，使用余弦相似度衡量数字之间的相似程度。左图为相似度矩阵，右图为差异度矩阵。")
            ]}),
            new Paragraph({ alignment: AlignmentType.CENTER, children: [
                new ImageRun({
                    type: "png", data: images['06_digit_similarity.png'],
                    transformation: { width: 600, height: 225 },
                    altText: { title: "相似度", description: "数字相似度", name: "similarity" }
                })
            ]}),
            new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 120, after: 360 }, children: [
                new TextRun({ text: "图4.6 数字间相似度与差异度矩阵", italics: true, size: 20 })
            ]}),
            new Paragraph({ spacing: { after: 120 }, children: [
                new TextRun({ text: "分析发现：", bold: true })
            ]}),
            new Paragraph({ numbering: { reference: "bullet-list", level: 0 }, children: [
                new TextRun({ text: "数字4和9最相似（相似度0.917），最容易混淆", bold: true })
            ]}),
            new Paragraph({ numbering: { reference: "bullet-list", level: 0 }, children: [
                new TextRun("数字1和0差异最大（最不相似）")
            ]}),
            new Paragraph({ numbering: { reference: "bullet-list", level: 0 }, children: [
                new TextRun("数字3、5、8彼此间相似度较高")
            ]}),

            // 4.7 特征重要性
            new Paragraph({ children: [new PageBreak()] }),
            new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun("4.7 特征重要性分析")] }),
            new Paragraph({ spacing: { after: 240 }, children: [
                new TextRun("使用随机森林分类器评估每个像素对数字分类的贡献度。左图展示所有像素的重要性，右图标注Top 100最重要像素的位置。")
            ]}),
            new Paragraph({ alignment: AlignmentType.CENTER, children: [
                new ImageRun({
                    type: "png", data: images['07_feature_importance.png'],
                    transformation: { width: 550, height: 275 },
                    altText: { title: "特征重要性", description: "随机森林特征重要性", name: "rf" }
                })
            ]}),
            new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 120, after: 360 }, children: [
                new TextRun({ text: "图4.7 随机森林特征重要性分析", italics: true, size: 20 })
            ]}),
            new Paragraph({ spacing: { after: 120 }, children: [
                new TextRun({ text: "核心结论：", bold: true })
            ]}),
            new Paragraph({ numbering: { reference: "bullet-list", level: 0 }, children: [
                new TextRun({ text: "中心区域像素对分类贡献最大", bold: true })
            ]}),
            new Paragraph({ numbering: { reference: "bullet-list", level: 0 }, children: [
                new TextRun("边缘4-6像素完全不重要，可以裁剪")
            ]}),
            new Paragraph({ numbering: { reference: "bullet-list", level: 0 }, children: [
                new TextRun("重要像素主要分布在数字笔画交叉和转折的关键位置")
            ]}),

            // 4.8 混淆矩阵
            new Paragraph({ children: [new PageBreak()] }),
            new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun("4.8 分类混淆矩阵")] }),
            new Paragraph({ spacing: { after: 240 }, children: [
                new TextRun("使用随机森林分类器对10,000个样本进行分类，生成混淆矩阵评估分类性能。左图为绝对数量，右图为归一化比例。")
            ]}),
            new Paragraph({ alignment: AlignmentType.CENTER, children: [
                new ImageRun({
                    type: "png", data: images['08_confusion_matrix.png'],
                    transformation: { width: 600, height: 225 },
                    altText: { title: "混淆矩阵", description: "分类混淆矩阵", name: "confusion" }
                })
            ]}),
            new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 120, after: 360 }, children: [
                new TextRun({ text: "图4.8 分类混淆矩阵（随机森林，准确率100%）", italics: true, size: 20 })
            ]}),
            new Paragraph({ spacing: { after: 120 }, children: [
                new TextRun({ text: "性能分析：", bold: true })
            ]}),
            new Paragraph({ numbering: { reference: "bullet-list", level: 0 }, children: [
                new TextRun("训练集准确率达到100%（可能存在过拟合）")
            ]}),
            new Paragraph({ numbering: { reference: "bullet-list", level: 0 }, children: [
                new TextRun("对角线数值高，说明模型对各数字识别性能均衡")
            ]}),

            // 4.9 聚类分析
            new Paragraph({ children: [new PageBreak()] }),
            new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun("4.9 K-means聚类分析")] }),
            new Paragraph({ spacing: { after: 240 }, children: [
                new TextRun("使用K-means算法（K=10）对手写数字进行无监督聚类。左图按真实标签着色，右图按聚类结果着色，对比两者可评估聚类质量。")
            ]}),
            new Paragraph({ alignment: AlignmentType.CENTER, children: [
                new ImageRun({
                    type: "png", data: images['09_clustering_analysis.png'],
                    transformation: { width: 600, height: 225 },
                    altText: { title: "聚类", description: "K-means聚类", name: "clustering" }
                })
            ]}),
            new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 120, after: 360 }, children: [
                new TextRun({ text: "图4.9 K-means聚类分析（K=10）", italics: true, size: 20 })
            ]}),
            new Paragraph({ spacing: { after: 120 }, children: [
                new TextRun({ text: "聚类结果：", bold: true })
            ]}),
            new Paragraph({ numbering: { reference: "bullet-list", level: 0 }, children: [
                new TextRun("聚类中心（红色X标记）大致对应10个数字的位置")
            ]}),
            new Paragraph({ numbering: { reference: "bullet-list", level: 0 }, children: [
                new TextRun("某些数字可被无监督聚类很好地分离（如0、1）")
            ]}),
            new Paragraph({ numbering: { reference: "bullet-list", level: 0 }, children: [
                new TextRun("数字4、7、9的聚类效果较差，需要监督学习")
            ]}),

            // 第五章：分析结论
            new Paragraph({ children: [new PageBreak()] }),
            new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun("5. 可视化分析结论")] }),
            new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun("5.1 数据特征总结")] }),
            new Paragraph({ spacing: { after: 120 }, children: [
                new TextRun("通过9种可视化技术的综合分析，我们对MNIST数据集有了深入理解：")
            ]}),
            new Paragraph({ numbering: { reference: "numbered-list", level: 0 }, children: [
                new TextRun({ text: "特征分布：", bold: true }),
                new TextRun("像素值呈双峰分布，大部分为背景（0值），中心区域包含主要信息，边缘4-6像素可裁剪。")
            ]}),
            new Paragraph({ numbering: { reference: "numbered-list", level: 0 }, children: [
                new TextRun({ text: "降维潜力：", bold: true }),
                new TextRun("283个主成分可保留95%信息（降维64%），说明存在大量冗余特征。")
            ]}),
            new Paragraph({ numbering: { reference: "numbered-list", level: 0 }, children: [
                new TextRun({ text: "类别差异：", bold: true }),
                new TextRun("数字0、1结构简单易识别；数字4、9最相似（0.917）易混淆；数字3、5、8彼此相似。")
            ]}),
            new Paragraph({ numbering: { reference: "numbered-list", level: 0 }, children: [
                new TextRun({ text: "非线性结构：", bold: true }),
                new TextRun("PCA效果有限（前2维仅10%方差），t-SNE更好地揭示聚类结构，说明数据高度非线性。")
            ]}),
            new Paragraph({ numbering: { reference: "numbered-list", level: 0 }, children: [
                new TextRun({ text: "分类性能：", bold: true }),
                new TextRun("随机森林在训练集达100%准确率，中心区域像素是关键特征。")
            ]}),

            new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun("5.2 识别难点分析")] }),
            new Paragraph({ spacing: { after: 120 }, children: [
                new TextRun("基于可视化分析，识别出以下挑战：")
            ]}),
            new Paragraph({ numbering: { reference: "bullet-list", level: 0 }, children: [
                new TextRun({ text: "类内变异：", bold: true }),
                new TextRun("同一数字的不同书写风格、笔画粗细、倾斜角度差异大，特别是4、5、7、9。")
            ]}),
            new Paragraph({ numbering: { reference: "bullet-list", level: 0 }, children: [
                new TextRun({ text: "类间相似：", bold: true }),
                new TextRun("数字4和9、3和5、3和8形状相近，容易混淆。")
            ]}),
            new Paragraph({ numbering: { reference: "bullet-list", level: 0 }, children: [
                new TextRun({ text: "样本质量：", bold: true }),
                new TextRun("部分样本存在位置偏移、笔画断裂等问题。")
            ]}),

            new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun("5.3 优化建议")] }),
            new Paragraph({ numbering: { reference: "numbered-list", level: 0 }, children: [
                new TextRun({ text: "数据增强：", bold: true }),
                new TextRun("对训练数据进行旋转、缩放、位移等变换，提高模型鲁棒性。")
            ]}),
            new Paragraph({ numbering: { reference: "numbered-list", level: 0 }, children: [
                new TextRun({ text: "特征工程：", bold: true }),
                new TextRun("可裁剪边缘4-6像素，从28×28降到20×20（降维51%）。")
            ]}),
            new Paragraph({ numbering: { reference: "numbered-list", level: 0 }, children: [
                new TextRun({ text: "模型选择：", bold: true }),
                new TextRun("考虑卷积神经网络(CNN)更好地捕捉图像的空间结构特征。")
            ]}),
            new Paragraph({ numbering: { reference: "numbered-list", level: 0 }, children: [
                new TextRun({ text: "难例挖掘：", bold: true }),
                new TextRun("针对4、9等易混淆数字进行专门训练。")
            ]}),

            // 第六章：总结
            new Paragraph({ children: [new PageBreak()] }),
            new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun("6. 总结与展望")] }),
            new Paragraph({ spacing: { after: 120 }, children: [
                new TextRun("本报告对Kaggle公开的MNIST手写数字识别数据集（70,000样本×784特征）进行了全面的可视化分析。通过9种不同的可视化技术，从多个角度深入探讨了手写数字的特征分布、类别可分性、降维效果和分类难点。")
            ]}),
            new Paragraph({ spacing: { after: 120 }, children: [
                new TextRun({ text: "核心成果：", bold: true })
            ]}),
            new Paragraph({ numbering: { reference: "bullet-list", level: 0 }, children: [
                new TextRun("识别出中心区域像素是关键特征，边缘可裁剪")
            ]}),
            new Paragraph({ numbering: { reference: "bullet-list", level: 0 }, children: [
                new TextRun("发现数字4和9最易混淆（相似度0.917）")
            ]}),
            new Paragraph({ numbering: { reference: "bullet-list", level: 0 }, children: [
                new TextRun("证明降维可保留大部分信息（283维→95%方差）")
            ]}),
            new Paragraph({ numbering: { reference: "bullet-list", level: 0 }, children: [
                new TextRun("揭示数据的非线性流形结构")
            ]}),
            new Paragraph({ spacing: { after: 120 }, children: [
                new TextRun("本研究展示了数据可视化在理解复杂数据集中的重要作用，为后续的模型设计、特征工程和性能优化提供了数据依据。未来可以扩展到其他图像识别任务，如Fashion-MNIST、CIFAR-10等更复杂的数据集。")
            ]}),

            new Paragraph({ spacing: { before: 360, after: 0 }, alignment: AlignmentType.CENTER, children: [
                new TextRun({ text: "--- 报告完 ---", bold: true, size: 28, color: "7F8C8D" })
            ]})
        ]
    }]
});

// 生成并保存文档
Packer.toBuffer(doc).then(buffer => {
    fs.writeFileSync("outputs/MNIST手写数字识别可视化分析报告.docx", buffer);
    console.log("\n✓ 报告文档生成成功！");
    console.log("  文件位置: outputs/MNIST手写数字识别可视化分析报告.docx");
    console.log("\n="*80);
    console.log("所有任务完成！");
    console.log("="*80);
}).catch(err => {
    console.error("生成报告时出错:", err);
});
