import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
"""
Real dong + Real jing:

"""
# 计算准确率
def accuracy(confusion):
    return np.sum(confusion.diagonal()) / np.sum(confusion)


def F1_score(confusion, average):
    # 计算每类的精确率
    Pc = confusion.diagonal() / np.sum(confusion, axis=1)
    # 计算每类的召回率
    Rc = confusion.diagonal() / np.sum(confusion, axis=0)
    # 计算F1_score
    f1 = 2*np.multiply(Pc, Rc)/(Pc + Rc)
    if average == "macro":
        f1 = np.sum(f1) / len(f1)
    elif average == "weighted":
        n = np.sum(confusion, axis=0)
        weighted = n / np.sum(n)
        f1 = np.sum(np.multiply(f1, weighted))
    return f1


if __name__ == "__main__":
    # 0: Real ART, 1: Real ART + Syn PV, 2: Real PV, 3: Real PV + Syn ART
    Confusion_matrix = {
        0: np.array([[107, 8, 3], [10, 99, 17], [2, 4, 49]]),
        1: np.array([[111, 6, 0], [8, 102, 5], [0, 3, 64]]),
        2: np.array([[88, 0, 8], [0, 85, 6], [10, 4, 54]]),
        3: np.array([[93, 0, 5], [0, 84, 5], [2, 2, 64]]),
        4: np.array([[78, 3, 1], [2, 67, 3], [4, 7, 59]]),
    }
    for j in range(len(Confusion_matrix)):
        acc = round(accuracy(Confusion_matrix[j]), 4) * 100
        f1_score = round(F1_score(Confusion_matrix[j], average="macro"), 4) * 100
        print(f"{j}——ACC:{acc}\t F1_score:{f1_score}")
    data_dir = "/home/lzp/project/HGDiff/Exp_Classify/test_acc/"
    fig, ax = plt.subplots(figsize=(6, 5), dpi=200)
    ax.grid(ls="--", lw=0.25, color="#4E616C")
    for k in sorted(os.listdir(data_dir)):
        # 绘制test对比曲线
        data = pd.read_csv(os.path.join(data_dir, k), sep=',', header='infer', usecols=[1])
        epoch_acc = data["test_acc"].to_dict()
        epochs = []
        accs = []
        for i in range(len(epoch_acc)):
            if i % 20 == 0:
                epochs.append(i)
                accs.append(epoch_acc[i])
        m = make_interp_spline(epochs, accs)
        xs = np.linspace(0, 100, 101)
        ys = m(xs)
        if k.split("_")[0] == "1":
            ax.plot(xs, ys, c='#AE2012', label=f"Real ART (ACC={accs[-1]})", linewidth=3, zorder=2)
            type1 = ax.scatter(epochs, accs, c='#AE2012', marker="o", s=50, zorder=2)
        elif k.split("_")[0] == "2":
            ax.plot(xs, ys, c='#EE9B00', label=f"Real PV (ACC={accs[-1]})", linewidth=3, zorder=2)
            ax.scatter(epochs, accs, c='#EE9B00', marker="x", s=50, zorder=2)
        elif k.split("_")[0] == "3":
            ax.plot(xs, ys, c='#E9DBA6', label=f"Real ART + Real PV (ACC={accs[-1]})", linewidth=3, zorder=2)
            ax.scatter(epochs, accs, c='#E9DBA6', marker="v", s=50, zorder=2)
        elif k.split("_")[0] == "4":
            ax.plot(xs, ys, c='#0A9396', label=f"Real ART + Syn PV (ACC={accs[-1]})", linewidth=3, zorder=2)
            ax.scatter(epochs, accs, c='#0A9396', marker=",", s=50, zorder=2)
        elif k.split("_")[0] == "5":
            ax.plot(xs, ys, c='#005F73', label=f"Real PV + Syn ART (ACC={accs[-1]})", linewidth=3, zorder=2)
            ax.scatter(epochs, accs, c='#005F73', marker="*", s=50, zorder=2)

    plt.xlabel("Epoch", fontdict={'size': 13})
    plt.ylabel("Accuracy", fontdict={'size': 13})
    ax.legend(loc=4, fontsize="small")
    plt.savefig("/home/lzp/project/HGDiff/Exp_Classify/test_acc.png", dpi=150)
    plt.show()





