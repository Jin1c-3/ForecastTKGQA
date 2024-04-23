from torch.utils.data import DataLoader
from collections import defaultdict
import numpy as np
from tqdm import tqdm
import pickle
import os
import torch.multiprocessing
import time
torch.multiprocessing.set_sharing_strategy('file_system')
import torch.distributed


def eval_model(qa_model, dataloader, dataset, split, batch_size, device): # Evaluation code
    with open('data/ICEWS21/filter_dict.pkl', 'rb') as f:
        filter_dict = pickle.load(f)
    qa_model.eval()

    total_loss = 0
    eval_log = []
    eval_log.append("Split %s" % (split))
    eval_log.append("Evaluation Result.")
    print('Evaluating split', split)

    num_questions_count = 0
    # For evaluating Hits@k
    mrr = 0
    mrr_1_hop = 0
    count_1_hop = 0
    mrr_2_hop = 0
    count_2_hop = 0
    predicted_answers = []
    truth_entities = []
    hop_types = []

    # For evaluating yes-no accuracy
    predicted_class = []
    truth_class = []
    score_clf = 0

    # For evaluating fact reasoning accuracy
    predicted_choice = []
    truth_choice = []
    score_choice = 0

    # Dataloader
    loader = tqdm(dataloader, total=len(dataloader), unit="batches")

    with torch.no_grad():
        for i_batch, (b_input_id, b_attention_mask, heads, tails, times, types, answers_single, batch_sentences) in enumerate(loader):
            question_tokenized = b_input_id.to(device)
            question_attention_mask = b_attention_mask.to(device)
            heads = heads.to(device)
            tails = tails.to(device)
            times = times.to(device)
            types = types.to(device)
            answers = answers_single.to(device)
            if i_batch * batch_size == len(dataset.data):
                break
            mask_dis, scores_ep, scores_yn, scores_mc = qa_model.forward(question_tokenized,
                                                                               question_attention_mask, heads, tails,
                                                                               times,
                                                                               types, answers)
            criterion = torch.nn.CrossEntropyLoss(reduction='mean').cuda(device)

            # 创建一个列表来存储所有的得分
            scores = [scores_ep, scores_yn, scores_mc]

            # 计算None的数量
            none_count = scores.count(None)

            # 如果None的数量小于等于1，或者全是None，引发一个异常
            if none_count <= 1 or none_count == 3:
                raise ValueError(
                    f"得分异常，{scores_ep=}, {scores_yn=}, {scores_mc=}")

            # 如果只有一个值不为None，将其存储在score变量中
            elif none_count == 2:
                score = next(s for s in scores if s is not None)
                mask = types < 2 if score is scores_ep else types == 2 if score is scores_yn else types == 3
                loss = criterion(score, answers[mask])
            else:
                raise ValueError("所有得分都不为None，无法确定唯一的得分")
            total_loss += loss.item()
            if mask_dis is not None:
                answers = answers[mask_dis]
                types = types[mask_dis]
            torch.distributed.barrier()
            if scores_ep is not None: # If the question type is entity prediction
                ind_batch = torch.arange(scores_ep.shape[0]).cuda(device)
                score_vector = scores_ep[ind_batch, answers.squeeze()].clone()  # scores for right answer
                filter = torch.zeros_like(scores_ep)
                filter_idx = 0  # from 0 to 255
                # assert 0, len(a[8])
                for text_idx in batch_sentences:  # a[-1] sentences list for indexing
                    # assert filter_idx < 256, text_idx
                    answers_multiple = filter_dict[text_idx][-1]
                    filter[filter_idx, answers_multiple] = 0.5  # index data in batch, change data in corresponding position from zero to a half
                    filter_idx += 1
                # assert 0, [filter.shape, filter.sum()]
                mask_ep = types < 2
                answers_ep = answers[mask_ep].tolist()
                types_ep = types[mask_ep].tolist()
                truth_entities.extend(answers_ep)
                hop_types.extend(dataset.ids_to_types(types_ep))
                # assert 0, scores_ep.dtype
                scores_ep = torch.where(filter > 0, torch.tensor(-100000.).cuda(device), scores_ep)
                scores_ep[ind_batch, answers.squeeze()] = score_vector
                for i, s in enumerate(scores_ep):
                    current_mrr = dataset.get_rank_from_scores(s, answers_ep[i])
                    mrr += current_mrr
                    current_type = types_ep[i]
                    if current_type == 0:
                        mrr_1_hop += current_mrr # 1-hop MRR
                        count_1_hop += 1
                    elif current_type == 1:
                        mrr_2_hop += current_mrr # 2-hop MRR
                        count_2_hop += 1
                    else:
                        raise ValueError('Wrong question type')
                    topk_answers = dataset.get_answers_from_scores(s, k=10)
                    predicted_answers.append(topk_answers)
            if scores_yn is not None: # If the question type is yes-no
                mask_yn = types == 2
                answers_yn = answers[mask_yn]
                truth_class.append(answers_yn)
                predicted_class.append(scores_yn.detach().cpu().numpy())
            if scores_mc is not None: # If the question type is fact reasoning
                mask_mc = types == 3
                answers_mc = answers[mask_mc]
                truth_choice.append(answers_mc)
                predicted_choice.append(scores_mc.detach().cpu().numpy())

    eval_log.append('Loss %f' % total_loss)
    eval_log.append('Eval batch size %d' % batch_size)

    if truth_entities: # If the question type is entity prediction
        # MRR computation
        mrr = mrr / len(truth_entities)
        eval_log.append('MRR: %f' % (round(mrr, 3)))
        if count_1_hop > 0:
            mrr_1_hop = mrr_1_hop / count_1_hop
            eval_log.append('MRR for 1-hop: %f' % (round(mrr_1_hop, 3)))
        if count_2_hop > 0:
            mrr_2_hop = mrr_2_hop / count_2_hop
            eval_log.append('MRR for 2-hop: %f' % (round(mrr_2_hop, 3)))
        eval_hits(truth_entities, hop_types, predicted_answers, eval_log) # Hits@k computation
    if truth_class: # If the question type is yes-no
        score_clf = eval_acc(predicted_class, truth_class, 'yes_no', eval_log) # Accuracy computation
    if truth_choice: # If the question type is fact reasoning
        score_choice = eval_acc(predicted_choice, truth_choice, 'multiple_choice', eval_log) # Accuracy computation

    # Print eval log and return it
    for s in eval_log:
        print(s)

    return mrr, score_clf, score_choice, eval_log


def eval_hits(truth_entities, hop_types, predicted_answers, eval_log): # Hits@k computation
    question2hits1, question2hits10 = defaultdict(list), defaultdict(list)
    hits1, hits10 = 0, 0
    total = len(truth_entities)
    for i, question in enumerate(zip(truth_entities, hop_types)):
        actual_answers = question[0]
        question_type = question[1]
        predicted = predicted_answers[i]
        top_1 = predicted[:1]
        top_10 = predicted[:10]
        if len({actual_answers}.intersection(set(top_1))) > 0: # Hits@1
            hits1 += 1
            val_hits1 = 1
        else:
            val_hits1 = 0
        question2hits1[question_type].append(val_hits1)
        if len({actual_answers}.intersection(set(top_10))) > 0: # Hits@10
            hits10 += 1
            val_hits10 = 1
        else:
            val_hits10 = 0
        question2hits10[question_type].append(val_hits10)

    # Total Hits@1
    hits1_total = hits1 / total
    eval_log.append('Hits at 1: %f' % (round(hits1_total, 3)))
    record_hits(question2hits1, eval_log)
    # Total Hits@10
    hits10_total = hits10 / total
    eval_log.append('Hits at 10: %f' % (round(hits10_total, 3)))
    record_hits(question2hits10, eval_log)


def eval_acc(predicts, truths, question_type, eval_log): # Accuracy computation
    prediction = np.concatenate(predicts, axis=0)
    prediction = np.argmax(prediction, axis=1).flatten() # Find the candidate with the highest score
    truth = np.concatenate([t.cpu() for t in truths], )
    test_accuracy = (sum(np.array(prediction) == np.array(truth)) / float(len(truth)))
    eval_log.append(f'accuracy for {question_type} {round(test_accuracy, 5)} \t total questions: {len(truth)}')

    return test_accuracy


def record_hits(results, eval_log):
    results = dict(sorted(results.items(), key=lambda x: x[0].lower()))
    for key, value in results.items():
        hits_at_k = sum(value) / len(value)
        s = '{q_type} \t {hits_at_k} \t total questions: {num_questions}'.format(
            q_type=key,
            hits_at_k=round(hits_at_k, 3),
            num_questions=len(value)
        )
        eval_log.append(s)
