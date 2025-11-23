#!/bin/bash

echo "========================================"
echo "MNIST可视化分析项目 - 环境配置脚本"
echo "MNIST Visualization Analysis - Setup"
echo "========================================"

# 检查Python环境
echo ""
echo "[1/5] 检查Python环境..."
if ! command -v python3 &> /dev/null; then
    echo "❌ 错误: 未找到Python 3"
    echo "请先安装Python 3.8或更高版本"
    exit 1
fi
PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
echo "✓ Python版本: $PYTHON_VERSION"

# 检查Node.js环境
echo ""
echo "[2/5] 检查Node.js环境..."
if ! command -v node &> /dev/null; then
    echo "⚠️  警告: 未找到Node.js"
    echo "报告生成功能将不可用，但可以运行数据分析"
    NODE_AVAILABLE=false
else
    NODE_VERSION=$(node --version)
    echo "✓ Node.js版本: $NODE_VERSION"
    NODE_AVAILABLE=true
fi

# 解压数据集
echo ""
echo "[3/5] 准备MNIST数据集..."
if [ -f "mnist_data/archive.zip" ]; then
    if [ ! -f "mnist_data/mnist_train.csv" ] || [ ! -f "mnist_data/mnist_test.csv" ]; then
        echo "正在解压 archive.zip..."
        unzip -o mnist_data/archive.zip -d mnist_data/
        echo "✓ 数据集解压完成"
    else
        echo "✓ 数据集已存在，跳过解压"
    fi
else
    echo "⚠️  警告: 未找到 mnist_data/archive.zip"
    echo "请将MNIST数据集的ZIP文件放置在 mnist_data/ 目录下"
    echo "下载地址: https://www.kaggle.com/datasets/oddrationale/mnist-in-csv"

    if [ ! -f "mnist_data/mnist_train.csv" ] || [ ! -f "mnist_data/mnist_test.csv" ]; then
        echo "❌ 错误: 缺少必要的数据文件"
        exit 1
    else
        echo "✓ 但数据文件已存在，继续执行"
    fi
fi

# 安装Python依赖
echo ""
echo "[4/5] 安装Python依赖..."
if [ -f "requirements.txt" ]; then
    pip3 install -r requirements.txt
    echo "✓ Python依赖安装完成"
else
    echo "❌ 错误: 未找到 requirements.txt"
    exit 1
fi

# 安装Node.js依赖
echo ""
echo "[5/5] 安装Node.js依赖..."
if [ "$NODE_AVAILABLE" = true ]; then
    if [ -f "package.json" ]; then
        npm install
        echo "✓ Node.js依赖安装完成"
    else
        echo "❌ 错误: 未找到 package.json"
        exit 1
    fi
else
    echo "⊘ 跳过Node.js依赖安装（Node.js不可用）"
fi

# 创建输出目录
echo ""
echo "创建输出目录..."
mkdir -p mnist_visualizations
mkdir -p outputs
echo "✓ 输出目录已创建"

echo ""
echo "========================================"
echo "✅ 环境配置完成！"
echo "========================================"
echo ""
echo "现在可以运行分析："
echo "  • 运行Python分析:  python3 data_analysis_and_visualization.py"
if [ "$NODE_AVAILABLE" = true ]; then
    echo "  • 生成Word报告:    node generate_mnist_report.js"
    echo "  • 一键完成:        npm run full"
fi
echo ""
