# coding=utf-8
import json
import sys
import re
import numpy as np
from text2vec import SentenceModel, semantic_search, Similarity

simModel_path = './pre_train_model/text2vec-base-chinese'  # 相似度模型路径
simModel = SentenceModel(model_name_or_path=simModel_path, device='cuda:0')


def calc_jaccard(list_a, list_b, threshold=0.3):
    size_a, size_b = len(list_a), len(list_b)
    list_c = [i for i in list_a if i in list_b]
    size_c = len(list_c)
    score = size_c / (size_b + 1e-6)
    if score > threshold:
        return 1
    else:
        return 0

def report_score(gold_path, predict_path):
    gold_info = json.load(open(gold_path))
    pred_info = json.load(open(predict_path))

    idx = 0
    for gold, pred in zip(gold_info, pred_info):
        question = gold["question"]
        keywords = gold["keywords"]
        gold_answer = gold["answer"].strip()

        # 找出 pred 中所有 answer_x
        answer_keys = [k for k in pred.keys() if k.startswith("answer_")]

        best_score = -1
        best_pred = ""
        best_key = None
        
        for ans_key in answer_keys:
            pred_answer = pred[ans_key].strip()

            # --- Score 计算方式不变 ---
            if gold_answer == "无答案" and pred_answer != gold_answer:
                score = 0.0
            elif gold_answer == "无答案" and pred_answer == gold_answer:
                score = 1.0
            else:
                semantic_score = semantic_search(
                    simModel.encode([gold_answer]),
                    simModel.encode(pred_answer),
                    top_k=1
                )[0][0]["score"]

                join_keywords = [word for word in keywords if word in pred_answer]
                keyword_score = calc_jaccard(join_keywords, keywords)

                score = 0.5 * keyword_score + 0.5 * semantic_score
            
            # --- 保留最高分答案 ---
            # if score > best_score and int(ans_key[-1]) <= 3:
            if score > best_score:
                best_score = score
                best_pred = pred_answer
                best_key = ans_key
            

        # 保存最终 score 和选择的答案
        gold_info[idx]["score"] = best_score
        gold_info[idx]["predict"] = best_pred
        gold_info[idx]["predict_from"] = best_key  # 可选，说明是 answer_几
        
        print(f"Predict: {question}, From {best_key}, Score: {best_score}")

        idx += 1

    return gold_info


if __name__ == "__main__":
    '''
      online evaluation
    '''

    # 标准答案路径
    gold_path = "./data/gold.json"
    print("Read gold from %s" % gold_path)

    # 预测文件路径
    predict_path = "./data/result_our_new.json"
    print("Read predict file from %s" % predict_path)

    results = report_score(gold_path, predict_path)

    # 输出最终得分
    final_score = np.mean([item["score"] for item in results])
    print("\n")
    print("=" * 100)
    print(f"Total number of predicted questions: {len(results)}, Final score: {final_score}")
    print("=" * 100)

    # 结果文件路径
    metric_path = "./data/metrics.json"
    output_json = {
    "final_score": final_score,
    "results": results
    }
    
    results_info = json.dumps(output_json, ensure_ascii=False, indent=2)
    with open(metric_path, "w", encoding="utf-8") as fd:
        fd.write(results_info)
    print(f"\nsave json file to {metric_path}")
