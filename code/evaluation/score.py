import json
import math
import collections

def eval(pred, gold):

    with open(gold, 'r', errors='ignore') as f:
        goldjson = json.load(f, strict=False)

    if type(pred) != type('123'):
        outputjson = pred
    else:
        with open(pred, 'r', errors='ignore') as f:
            outputjson = json.load(f, strict=False)

    def NDCG(pred, gold):

        # Discounted Cumulative Gain for predictions
        pred_score = [gold.get(i,0) for i in pred]
        DCG_pred, idx = 0, 1
        for i, c in enumerate(pred_score):
            idx += 1
            DCG_pred += c / math.log(idx, 2)

        # Discounted Cumulative Gain for the ground truth
        gold = sorted(gold.values(), reverse=True) + [0] * len(pred)
        DCG_gold, idx = 0, 1
        for i, c in enumerate(pred_score):
            idx += 1
            DCG_gold += gold[i] / math.log(idx, 2)

        res = DCG_pred/DCG_gold

        return res

    detection_tp, detection_pred, detection_gold = 0, 0, 0
    detection_tp_level, detection_gold_level = collections.defaultdict(int), collections.defaultdict(int)
    detection_tp_type, detection_gold_type= [0, 0, 0], [0, 0, 0]
    weighted_detection_tp, weighted_detection_gold = 0, 0
    recommendation_tp, recommendation_pred, recommendation_gold = 0, 0, 0
    e2e_tp, e2e_pred, e2e_gold = 0, 0, 0
    ndcg, recommendation_acc = 0, 0
    substitute_word_num, total_word_num, gold_substitute_word_num = 0, 0, 0

    e2e_tp_type, e2e_gold_type = [0, 0, 0], [0, 0, 0]
    recommendation_acc_type = [0, 0, 0]

    for idx, v in goldjson.items():

        total_word_num += len(outputjson[idx]['input_words'])
        substitute_word_num += sum([i[0][2]-i[0][1] for i in outputjson[idx]['substitute_topk']])

        pred_res = {"%d:%d" % (k[1],k[2]):v for k,v in outputjson[idx]['substitute_topk']}
        gold_res = v['substitutes']
        gold_res_clip = {}
        gold_res_clip_type = [{}, {}]
        for k, vsub, typenum in gold_res: 
            thiskdict = {}
            for kk, vv in vsub.items():
                thiskdict[kk] = vv
                gold_res_clip["%d:%d" % (k[0],k[1])] = thiskdict
                gold_res_clip_type[typenum-1]["%d:%d" % (k[0],k[1])] = thiskdict
        gold_substitute_word_num += len(gold_res_clip)

        pred_detection = set(pred_res.keys())
        gold_detection = set(gold_res_clip.keys())

        improvable_target = {k:sum(v.values()) for k, v in gold_res_clip.items()}
        for k, v in improvable_target.items():
            detection_gold_level[v-1] += 1
            if k in pred_detection: detection_tp_level[v-1] += 1

        tp_substitute_num, pred_substitute_num, gold_substitute_num = 0, 0, 0
        tp_substitute_num_type, gold_substitute_num_type = [0, 0, 0], [0, 0, 0]
        for k, v in gold_res_clip.items():
            if k in pred_res:
                pred_substitute_num += len(pred_res[k])
                gold_substitute_num += len(v)
                tp_substitute_num += len(set(pred_res[k]) & set(v.keys()))
                ndcg += NDCG(pred_res[k], v)
                if pred_res[k][0] in v: recommendation_acc += 1; e2e_tp += 1
                # else: print(idx, k, v, pred_res[k])

        for typeid in range(2):
            for k, v in gold_res_clip_type[typeid].items():
                if k in pred_res:
                    gold_substitute_num_type[typeid] += len(v)
                    tp_substitute_num_type[typeid] += len(set(pred_res[k]) & set(v.keys()))
                    if pred_res[k][0] in v:
                        recommendation_acc_type[typeid] += 1; e2e_tp_type[typeid] += 1

        weighted_detection_dict = {k:sum(v.values()) for k, v in gold_res_clip.items()}

        tp_select = list(pred_detection & gold_detection)
        detection_tp += len(tp_select)
        detection_pred += len(pred_detection)
        detection_gold += len(gold_detection)
        
        for typeid in range(2):
            detection_tp_type[typeid] += len(list(pred_detection & set(gold_res_clip_type[typeid].keys())))
            detection_gold_type[typeid] += len(gold_res_clip_type[typeid].keys())
            e2e_gold_type[typeid] += len(gold_res_clip_type[typeid].keys())

        weighted_detection_tp += sum([weighted_detection_dict[i] for i in tp_select])
        weighted_detection_gold += sum(weighted_detection_dict.values())

        recommendation_tp += tp_substitute_num
        recommendation_pred += pred_substitute_num
        recommendation_gold += gold_substitute_num

        e2e_pred += len(pred_res)
        e2e_gold += len(gold_res_clip)

        # break

    p_detection = detection_tp / detection_pred
    r_detection = detection_tp / detection_gold
    r_detection_type = []
    for typeid in range(2):
        r_detection_type.append(detection_tp_type[typeid] / detection_gold_type[typeid])

    r_detection_level = {}
    for i in range(100):
        if detection_gold_level[i]: r_detection_level[i] = (detection_tp_level[i]/detection_gold_level[i]) 

    f_detection = 2 * p_detection * r_detection / (p_detection+r_detection)
    f_detection_05 = 1.25 * p_detection * r_detection / (0.25 * p_detection+r_detection)
    weighted_acc_detection = weighted_detection_tp/weighted_detection_gold

    p_recommendation = recommendation_tp / recommendation_pred
    r_recommendation = recommendation_tp / recommendation_gold
    f_recommendation = 2 * p_recommendation * r_recommendation / (p_recommendation + r_recommendation )
    f_recommendation_05 = 1.25 * p_recommendation * r_recommendation / (0.25 *p_recommendation + r_recommendation )
    ndcg_recommendation = ndcg/detection_tp
    acc_recommendation = recommendation_acc/detection_tp
    acc_recommendation_type = []
    for typeid in range(2):
        acc_recommendation_type.append(recommendation_acc_type[typeid] / detection_tp_type[typeid])

    p_e2e = e2e_tp / e2e_pred
    r_e2e = e2e_tp / e2e_gold

    r_e2e_type = []
    for typeid in range(2):
        r_e2e_type.append(e2e_tp_type[typeid] / e2e_gold_type[typeid])

    f_e2e = (2 * p_e2e * r_e2e / (p_e2e+r_e2e)) if p_e2e+r_e2e else 0
    f_e2e_05 = (1.25 * p_e2e * r_e2e / (0.25 *p_e2e+r_e2e)) if p_e2e+r_e2e else 0

    substitute_rate = substitute_word_num / total_word_num

    return {
        "p_detection": p_detection,
        "r_detection": r_detection,
        "r_detection_type": r_detection_type,
        "f_detection": f_detection,
        "f_detection_05": f_detection_05,
        "weighted_acc_detection": weighted_acc_detection,
        "p_recommendation": p_recommendation,
        "r_recommendation": r_recommendation,
        "f_recommendation": f_recommendation,
        "f_recommendation_05": f_recommendation_05,
        "ndcg_recommendation": ndcg_recommendation,
        "acc_recommendation": acc_recommendation,
        "acc_recommendation_type": acc_recommendation_type,
        "p_e2e": p_e2e, 
        "r_e2e": r_e2e,
        "r_e2e_type": r_e2e_type,
        "f_e2e": f_e2e,
        "f_e2e_05": f_e2e_05,
        "substitute_rate": substitute_rate
    }

if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", type=str, default='../../data/sws/sws_test.json')
    parser.add_argument("--pred", type=str, default="/home/v-chenswang/iven/code/WriteModelTraining/WordSubstitution/data/wiki/res_random_test.json")
    parser.add_argument("--name", type=str, default='score_example')
    args = parser.parse_args()

    res = eval(args.pred, args.gold)

    # print(json.dumps(res, indent=4))

    print("| %s | %.3f | %.3f | %.3f | %.3f | %.3f | %.3f | %.3f | %.3f | %.3f | %.3f |" % (
        args.name, 
        res['p_detection'], 
        res['r_detection'],
        res['f_detection_05'], 
        res['weighted_acc_detection'], 
        res['substitute_rate'], 
        res['ndcg_recommendation'], 
        res['acc_recommendation'], 
        res['p_e2e'], 
        res['r_e2e'], 
        res['f_e2e_05']
        ))
