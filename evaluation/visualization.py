import json
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
import numpy as np
import os

with open('./data/metrics_our_new_2.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

with open('./data/metrics_old_2.json', 'r', encoding='utf-8') as f:
    data2 = json.load(f)
    
print("Successfully open the json file!")

directory_path = "./evaluation/answer_1_to_7"

# ---------------------
# Score Histogram
# ---------------------
scores = [item["score"] for item in data]
scores2 = [item["score"] for item in data2]

plt.figure(figsize=(8, 6))
# plt.hist(scores, bins=20)
plt.hist([scores, scores2], bins=20, label=["Our Model", "Baseline"], alpha=0.7)
plt.xlabel("Score")
plt.ylabel("Count")
plt.title("Histogram of Scores")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(directory_path, "score_histogram_compare.png"), dpi=300)
plt.close()
print("draw Histogram!")


# ---------------------
# Predict_from Bar Chart 和 Pie Chart
# ---------------------
files = {
    "Our Model": "./data/metrics_our_new_2.json",
    "Baseline": "./data/metrics_old_2.json"
}
answer_order = ["answer_1", "answer_2", "answer_3", "answer_4", 
                "answer_5", "answer_6", "answer_7"]
# answer_order = ["answer_1", "answer_2", "answer_3"]

'''
sources = [item["predict_from"] for item in data 
           if item["predict_from"] in answer_order]

count = Counter(sources)

# 根据预设顺序取值
counts_ordered = [count[a] for a in answer_order]


# Bar chart
plt.figure(figsize=(8, 6))
plt.bar(answer_order, counts_ordered)
plt.xlabel("Predict From")
plt.ylabel("Count")
plt.title("Distribution of Selected Answers (Bar Chart)")
plt.tight_layout()
plt.savefig("./evaluation/answer_1_to_3/predict_from_bar.png", dpi=300)
plt.close()
print("draw bar!")
'''


# 统计每个 JSON 的 predict_from
counts_dict = {}
for name, path in files.items():
    with open(path,'r',encoding='utf-8') as f:
        data = json.load(f)
    sources = [item["predict_from"] for item in data if item["predict_from"] in answer_order]
    count = Counter(sources)
    counts_dict[name] = [count[a] for a in answer_order]  # 按顺序取值

# 并列柱状图参数
num_models = len(files)
x = np.arange(len(answer_order))
bar_width = 0.35  # 每个柱子的宽度

plt.figure(figsize=(10,6))

for i, (name, counts) in enumerate(counts_dict.items()):
    plt.bar(x + i*bar_width, counts, width=bar_width, label=name)

# X 轴居中显示
plt.xticks(x + bar_width*(num_models-1)/2, answer_order)

plt.xlabel("Predict From")
plt.ylabel("Count")
plt.title("Distribution of Selected Answers (Side-by-Side Bar Chart)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(directory_path, "predict_from_bar_compare.png"), dpi=300)
plt.close()
print("draw side-by-side bar chart!")


# Pie chart
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

for ax, (model_name, counts) in zip(axes, counts_dict.items()):
    ax.pie(counts, labels=answer_order, autopct='%1.1f%%', counterclock=True)
    ax.set_title(f"Distribution of Selected Answers ({model_name})", y=0.95)

plt.tight_layout()
plt.savefig(os.path.join(directory_path, "predict_from_pie_subplots.png"), dpi=300)
plt.close()

print("draw side-by-side pie charts!")


# 平均 score 折线图
plt.figure(figsize=(8, 6))

for name, path in files.items():
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 统计每个 answer 的 score 列表
    score_dict = defaultdict(list)
    for item in data:
        src = item["predict_from"]
        if src in answer_order:
            score_dict[src].append(item["score"])

    # 计算平均 score
    avg_scores = []
    for a in answer_order:
        if score_dict[a]:
            avg_scores.append(sum(score_dict[a])/len(score_dict[a]))
        else:
            avg_scores.append(0)

    # 画折线
    plt.plot(answer_order, avg_scores, marker='o', label=name)

plt.xlabel("Predict From")
plt.ylabel("Average Score")
plt.title("Average Score by Predict From (Multiple Models)")
plt.ylim(0, 1)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(directory_path, "avg_score_line_compare.png"), dpi=300)
plt.close()
print("draw avg score line plot for multiple JSONs!")


'''
# 平均 score 折线图
avg_scores = []
for a in answer_order:
    if score_dict[a]:  # 避免为空导致除 0
        avg_scores.append(sum(score_dict[a]) / len(score_dict[a]))
    else:
        avg_scores.append(0)

plt.figure(figsize=(8, 6))
plt.plot(answer_order, avg_scores, marker='o')  # marker='o' 更直观
plt.xlabel("Predict From")
plt.ylabel("Average Score")
plt.title("Average Score by Predict From")
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig("./evaluation/avg_score_line.png", dpi=300)
plt.close()
print("draw avg score line plot!")
'''
