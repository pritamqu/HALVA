import os
import json
import argparse
from collections import defaultdict


def get_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--data_path',
                        type=str,
                        default="/scratch/ssd004/scratch/pritam/datasets/MME/MME_Benchmark_release_version")
    
    parser.add_argument('--result_file',
                        type=str,
                        default="path/to/jsonl")

    args = parser.parse_args()
    return args


def get_gt(data_path):
    GT = {}
    for category in os.listdir(data_path):
        category_dir = os.path.join(data_path, category)
        if not os.path.isdir(category_dir):
            continue
        if os.path.exists(os.path.join(category_dir, 'images')):
            image_path = os.path.join(category_dir, 'images')
            qa_path = os.path.join(category_dir, 'questions_answers_YN')
        else:
            image_path = qa_path = category_dir
        assert os.path.isdir(image_path), image_path
        assert os.path.isdir(qa_path), qa_path
        for file in os.listdir(qa_path):
            if not file.endswith('.txt'):
                continue
            for line in open(os.path.join(qa_path, file)):
                question, answer = line.strip().split('\t')
                GT[(category, file, question)] = answer
    return GT

if __name__ == "__main__":

    args = get_args()

    GT = get_gt(
        data_path=args.data_path
    )

    answers = [json.loads(line) for line in open(args.result_file)]
    result_dir = os.path.dirname(args.result_file)
    result_dir = os.path.join(result_dir, 'answers')
    os.makedirs(result_dir, exist_ok=True)

    results = defaultdict(list)
    for answer in answers:
        category = answer['question_id'].split('/')[0]
        file = answer['question_id'].split('/')[-1].split('.')[0] + '.txt'
        question = answer['prompt']
        results[category].append((file, answer['prompt'], answer['text']))

    for category, cate_tups in results.items():
        with open(os.path.join(result_dir, f'{category}.txt'), 'w') as fp:
            for file, prompt, answer in cate_tups:
                if 'Answer the question using a single word or phrase.' in prompt:
                    prompt = prompt.replace('Answer the question using a single word or phrase.', '').strip()
                if 'Please answer yes or no.' not in prompt:
                    prompt = prompt + ' Please answer yes or no.'
                    if (category, file, prompt) not in GT:
                        prompt = prompt.replace(' Please answer yes or no.', '  Please answer yes or no.')
                gt_ans = GT[category, file, prompt]
                tup = file, prompt, gt_ans, answer
                fp.write('\t'.join(tup) + '\n')