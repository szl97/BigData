"""
推荐系统可视化脚本
Visualization Script for Recommendation System

可视化训练过程、评估结果和模型分析
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import os


def load_results(filename='experiment_results.json'):
    """
    加载实验结果

    参数:
        filename: 结果文件名

    返回:
        results: 结果字典
    """
    filepath = os.path.join(os.path.dirname(__file__), filename)

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Results file not found: {filepath}")

    with open(filepath, 'r', encoding='utf-8') as f:
        results = json.load(f)

    return results


def plot_training_curve(results, save_path='training_curve.png'):
    """
    绘制训练曲线

    参数:
        results: 实验结果字典
        save_path: 保存路径
    """
    plt.figure(figsize=(10, 6))

    rmse_history = results['training_history']['rmse_per_epoch']
    epochs = range(1, len(rmse_history) + 1)

    plt.plot(epochs, rmse_history, 'b-', linewidth=2, label='Training RMSE')

    # 添加目标线
    plt.axhline(y=0.85, color='r', linestyle='--', linewidth=2, label='Target (0.85)')

    # 添加测试集RMSE
    test_rmse = results['performance']['test_rmse']
    plt.axhline(y=test_rmse, color='g', linestyle='-.', linewidth=2,
                label=f'Test RMSE ({test_rmse:.4f})')

    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('RMSE', fontsize=12)
    plt.title('Training Curve - Hybrid Recommender System', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)

    # 设置y轴范围
    min_rmse = min(rmse_history)
    plt.ylim([min_rmse * 0.95, max(rmse_history[0], 0.90)])

    plt.tight_layout()

    output_path = os.path.join(os.path.dirname(__file__), save_path)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Training curve saved to: {output_path}")

    plt.close()


def plot_performance_comparison(results, save_path='performance_comparison.png'):
    """
    绘制性能对比图

    参数:
        results: 实验结果字典
        save_path: 保存路径
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # 子图1: RMSE对比
    train_rmse = results['performance']['train_rmse']
    test_rmse = results['performance']['test_rmse']
    baseline = 0.89
    target = 0.85

    categories = ['Baseline', 'Target', 'Train', 'Test']
    values = [baseline, target, train_rmse, test_rmse]
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']

    bars = ax1.bar(categories, values, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)

    # 添加数值标签
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.4f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax1.set_ylabel('RMSE', fontsize=12)
    ax1.set_title('RMSE Comparison', fontsize=14, fontweight='bold')
    ax1.set_ylim([0, max(values) * 1.1])
    ax1.grid(True, axis='y', alpha=0.3)

    # 子图2: 改进幅度
    improvement = results['performance']['improvement_from_baseline']
    improvement_pct = (improvement / baseline) * 100

    ax2.barh(['Improvement'], [improvement], color='#95E1D3', alpha=0.7,
             edgecolor='black', linewidth=1.5)

    ax2.text(improvement/2, 0, f'{improvement:.4f}\n({improvement_pct:.2f}%)',
            ha='center', va='center', fontsize=12, fontweight='bold')

    ax2.set_xlabel('RMSE Reduction', fontsize=12)
    ax2.set_title('Improvement from Baseline (0.89)', fontsize=14, fontweight='bold')
    ax2.set_xlim([0, improvement * 1.2])
    ax2.grid(True, axis='x', alpha=0.3)

    plt.tight_layout()

    output_path = os.path.join(os.path.dirname(__file__), save_path)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Performance comparison saved to: {output_path}")

    plt.close()


def plot_convergence_analysis(results, save_path='convergence_analysis.png'):
    """
    绘制收敛性分析

    参数:
        results: 实验结果字典
        save_path: 保存路径
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    rmse_history = results['training_history']['rmse_per_epoch']
    epochs = range(1, len(rmse_history) + 1)

    # 子图1: 对数尺度的训练曲线
    ax1.semilogy(epochs, rmse_history, 'b-', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('RMSE (log scale)', fontsize=12)
    ax1.set_title('Training Convergence (Log Scale)', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # 子图2: RMSE变化率
    rmse_changes = []
    for i in range(1, len(rmse_history)):
        change = abs(rmse_history[i] - rmse_history[i-1])
        rmse_changes.append(change)

    ax2.plot(range(2, len(rmse_history) + 1), rmse_changes, 'r-', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('|RMSE Change|', fontsize=12)
    ax2.set_title('Training Stability', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    output_path = os.path.join(os.path.dirname(__file__), save_path)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Convergence analysis saved to: {output_path}")

    plt.close()


def plot_summary_dashboard(results, save_path='summary_dashboard.png'):
    """
    绘制综合仪表板

    参数:
        results: 实验结果字典
        save_path: 保存路径
    """
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # 1. 训练曲线 (大图)
    ax1 = fig.add_subplot(gs[0:2, 0:2])
    rmse_history = results['training_history']['rmse_per_epoch']
    epochs = range(1, len(rmse_history) + 1)
    ax1.plot(epochs, rmse_history, 'b-', linewidth=2, label='Training RMSE')
    ax1.axhline(y=0.85, color='r', linestyle='--', linewidth=2, label='Target')
    test_rmse = results['performance']['test_rmse']
    ax1.axhline(y=test_rmse, color='g', linestyle='-.', linewidth=2,
                label=f'Test ({test_rmse:.4f})')
    ax1.set_xlabel('Epoch', fontsize=11)
    ax1.set_ylabel('RMSE', fontsize=11)
    ax1.set_title('Training Progress', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # 2. 性能对比
    ax2 = fig.add_subplot(gs[0, 2])
    train_rmse = results['performance']['train_rmse']
    categories = ['Train', 'Test']
    values = [train_rmse, test_rmse]
    colors = ['#45B7D1', '#96CEB4']
    bars = ax2.bar(categories, values, color=colors, alpha=0.7, edgecolor='black')
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax2.set_ylabel('RMSE', fontsize=11)
    ax2.set_title('Performance', fontsize=13, fontweight='bold')
    ax2.grid(True, axis='y', alpha=0.3)

    # 3. 数据集统计
    ax3 = fig.add_subplot(gs[1, 2])
    ax3.axis('off')
    dataset_info = results['dataset']
    info_text = f"""
    Dataset Statistics
    ──────────────────
    Users:     {dataset_info['n_users']}
    Items:     {dataset_info['n_items']}
    Ratings:   {dataset_info['n_ratings']}
    Density:   {dataset_info['density']*100:.2f}%

    Train:     {dataset_info['train_size']}
    Test:      {dataset_info['test_size']}
    """
    ax3.text(0.1, 0.5, info_text, fontsize=10, family='monospace',
            verticalalignment='center')

    # 4. 超参数
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.axis('off')
    hyperparams = results['hyperparameters']
    hp_text = f"""
    Hyperparameters
    ──────────────────
    Factors:    {hyperparams['n_factors']}
    Epochs:     {hyperparams['n_epochs']}
    Learn Rate: {hyperparams['learning_rate']}
    Reg User:   {hyperparams['reg_user']}
    Reg Item:   {hyperparams['reg_item']}
    Neighbors:  {hyperparams['k_neighbors']}
    """
    ax4.text(0.1, 0.5, hp_text, fontsize=10, family='monospace',
            verticalalignment='center')

    # 5. 时间统计
    ax5 = fig.add_subplot(gs[2, 1])
    time_info = results['time']
    train_time = time_info['training_time_seconds']
    total_time = time_info['total_time_seconds']
    labels = ['Training', 'Other']
    sizes = [train_time, total_time - train_time]
    colors_pie = ['#FF6B6B', '#FED766']
    ax5.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90,
           colors=colors_pie, textprops={'fontsize': 10})
    ax5.set_title('Time Distribution', fontsize=13, fontweight='bold')

    # 6. 目标达成状态
    ax6 = fig.add_subplot(gs[2, 2])
    ax6.axis('off')
    achieved = results['performance']['target_achieved']
    improvement = results['performance']['improvement_from_baseline']

    if achieved:
        status = "✅ TARGET ACHIEVED"
        color = 'green'
    else:
        status = "❌ NOT YET"
        color = 'red'

    status_text = f"""

    {status}

    Test RMSE: {test_rmse:.4f}
    Target:    0.8500
    Gap:       {abs(test_rmse - 0.85):.4f}

    Improvement: {improvement:.4f}
    """
    ax6.text(0.5, 0.5, status_text, fontsize=11, family='monospace',
            verticalalignment='center', horizontalalignment='center',
            bbox=dict(boxstyle='round', facecolor=color, alpha=0.2))

    # 总标题
    fig.suptitle('Hybrid Recommender System - Experiment Dashboard',
                fontsize=16, fontweight='bold', y=0.98)

    output_path = os.path.join(os.path.dirname(__file__), save_path)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Summary dashboard saved to: {output_path}")

    plt.close()


def main():
    """
    主函数: 生成所有可视化图表
    """
    print("\n" + "=" * 60)
    print("Recommendation System Visualization".center(60))
    print("=" * 60)

    try:
        # 加载结果
        print("\nLoading experiment results...")
        results = load_results('experiment_results.json')

        # 生成各种可视化
        print("\nGenerating visualizations...")
        print("-" * 60)

        plot_training_curve(results)
        plot_performance_comparison(results)
        plot_convergence_analysis(results)
        plot_summary_dashboard(results)

        print("-" * 60)
        print("\n✅ All visualizations generated successfully!")
        print("\nGenerated files:")
        print("  • training_curve.png - Training progress over epochs")
        print("  • performance_comparison.png - RMSE comparison and improvement")
        print("  • convergence_analysis.png - Convergence and stability analysis")
        print("  • summary_dashboard.png - Comprehensive dashboard")

    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\n💡 Please run 'python run_experiment.py' first to generate results.")

    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
