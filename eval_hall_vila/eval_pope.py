import os
import json
import argparse
import wandb

def eval_pope(answers, label_file):
    label_list = [json.loads(q)['label'] for q in open(label_file, 'r')]

    for answer in answers:
        text = answer['text']

        # Only keep the first sentence
        if text.find('.') != -1:
            text = text.split('.')[0]

        text = text.replace(',', '')
        words = text.split(' ')
        if 'No' in words or 'not' in words or 'no' in words:
            answer['text'] = 'no'
        else:
            answer['text'] = 'yes'

    for i in range(len(label_list)):
        if label_list[i] == 'no':
            label_list[i] = 0
        else:
            label_list[i] = 1

    pred_list = []
    for answer in answers:
        if answer['text'] == 'no':
            pred_list.append(0)
        else:
            pred_list.append(1)

    pos = 1
    neg = 0
    yes_ratio = pred_list.count(1) / len(pred_list)

    TP, TN, FP, FN = 0, 0, 0, 0
    for pred, label in zip(pred_list, label_list):
        if pred == pos and label == pos:
            TP += 1
        elif pred == pos and label == neg:
            FP += 1
        elif pred == neg and label == neg:
            TN += 1
        elif pred == neg and label == pos:
            FN += 1

    print('TP\tFP\tTN\tFN\t')
    print('{}\t{}\t{}\t{}'.format(TP, FP, TN, FN))

    precision = float(TP) / float(TP + FP)
    recall = float(TP) / float(TP + FN)
    f1 = 2*precision*recall / (precision + recall)
    acc = (TP + TN) / (TP + TN + FP + FN)
    print('Accuracy: {}'.format(acc))
    print('Precision: {}'.format(precision))
    print('Recall: {}'.format(recall))
    print('F1 score: {}'.format(f1))
    print('Yes ratio: {}'.format(yes_ratio))
    print('%.3f, %.3f, %.3f, %.3f, %.3f' % (f1, acc, precision, recall, yes_ratio) )

    return {
      'precision': round(precision, 4),
      'recall': round(recall, 4),
      'f1': round(f1, 4),
      'accuracy': round(acc, 4),
      'yes_proportion': round(yes_ratio, 4),
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation-dir", type=str, default="/fs01/home/pritam/pritam_ssd004/datasets/POPE/coco")
    parser.add_argument("--question-file", type=str, default="/fs01/home/pritam/pritam_ssd004/datasets/POPE/coco_pope_all.jsonl")
    parser.add_argument("--result-file", type=str, default="/fs01/home/pritam/pritam_ssd004/OUTPUTS/CKD/11824280/answer_liuhaotian_llava-v1.5-7b_coco.jsonl")
    args = parser.parse_args()

    ans_file = args.result_file.split('/')[-1].split('_')
    dir_name = args.result_file.split('/')[-2]
    model_name = '_'.join(ans_file[1:-1])
    log_dir = '/'.join(args.result_file.split('/')[:-1])
    db=args.question_file.split('/')[-1].split('_')[0]

    print('wandb_id', dir_name+'_pope_'+db)
    wandb_writter = wandb.init(project=f"CKD-eval",
                                entity='pritamqu',
                                config={**vars(args), 'model_name':model_name, 'dir_name': dir_name, 'db': db},
                                dir=log_dir,
                                id=dir_name[:64]+'_pope_'+db,
                                name=dir_name[:64]+'_pope_'+db,
                                resume=True,
                                )

    questions = [json.loads(line) for line in open(args.question_file)]
    questions = {question['question_id']: question for question in questions}
    answers = [json.loads(q) for q in open(args.result_file)]
    avg_f1=0
    for file in os.listdir(args.annotation_dir):
        assert 'pope' in file
        assert file.endswith('.json')
        category = file.split('_')[-1][:-5]
        cur_answers = [x for x in answers if questions[x['question_id']]['category'] == category]
        print('Category: {}, # samples: {}'.format(category, len(cur_answers)))
        ans_dict = eval_pope(cur_answers, os.path.join(args.annotation_dir, file))

        for ans in ans_dict:
            wandb_writter.log({f'{category}/{ans}': ans_dict[ans], 'custom_step': 1})
            if ans=='f1':
                avg_f1+=ans_dict[ans]
            
        print("====================================")

    wandb_writter.log({f'pope/avg_f1': avg_f1/3, 'custom_step': 1})