'''
adopted from: https://github.com/tianyi-lab/HallusionBench/blob/main/random_guess.py
'''

import csv
import json
from tqdm import tqdm
import numpy as np
from prettytable import PrettyTable
import os
import time
from .utils import *
import random
import openai

from vila.utils import disable_torch_init
from vila.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path, is_gemma_tokenizer, KeywordsStoppingCriteria

import torch
from vila.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from vila.conversation import conv_templates, SeparatorStyle
from vila.model.builder_halva import load_pretrained_model
from PIL import Image

# load_json = False
load_json = True
pwd="eval_hall/hallusion_bench"
input_file_name = pwd+"/HallusionBench.json"
model_output_entry = "model_prediction"
model_correctness_entry = "gpt4v_output_gpt_check"


def load_model(model_path, model_base):

    # load model
    disable_torch_init()
    model_path = os.path.expanduser(model_path)
    model_name = get_model_name_from_path(model_path)

    print('Loading LLaVA...')
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, model_base=args.model_base, model_name=model_name)
    
    return tokenizer, model, image_processor


def generate_answer(data, model_output_entry):

    tokenizer, model, image_processor = load_model(model_path, model_base)

    # addl_prompt=" Please answer with one word Yes or No."
    addl_prompt=""
    for i in tqdm(data):

        qs = i["question"]+addl_prompt
        # print('INPUT: ', qs)
        image_file=None
        if int(i['visual_input'])>0:
            image_file = i["filename"][2:]
            if model.config.mm_use_im_start_end:
                qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        # print(f'input: {prompt}')

        if image_file is not None:
            image = Image.open(os.path.join(image_folder, image_file)).convert('RGB')
            image_tensor = process_images([image], image_processor, model.config)[0]

        # input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
        # input_ids = input_ids.to(device='cuda', non_blocking=True)
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        
        # conv = conv_templates[args.conv_mode]
        keywords = [conv.sep]
        stopping_criteria = [KeywordsStoppingCriteria(keywords, tokenizer, input_ids)] if args.conv_mode == "v0" or is_gemma_tokenizer(tokenizer) else None

        temperature=0
        top_p=None
        num_beams=1
        max_new_tokens=1024
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0).half().cuda() if image_file is not None else None,
                do_sample=True if temperature > 0 else False,
                temperature=temperature,
                top_p=top_p,
                num_beams=num_beams,
                max_new_tokens=max_new_tokens,
                use_cache=True, 
                stopping_criteria=stopping_criteria,
                )

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
        outputs = outputs.strip()

        # print('OUTPUT: ', outputs)

        i[model_output_entry] = outputs

    return data



# def generate_answer(data, model_output_entry):

#     for i in data:
#         i[model_output_entry] = "Yes" if random.random() > 0.5 else "No"

#     ## TODO
#     ## implement this section with yout model!
#     ## your_function(img_filename, question) -> "0" (No), "1" (Yes), "2" (Uncertain)
#     # for r in data:
#         # r[model_output_entry] = your_function(r["filename"], r["question"])

#     return data


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="liuhaotian/llava-v1.5-13b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="/fs01/home/pritam/pritam_ssd004/datasets/hallusion_bench")
    parser.add_argument("--save_json_path_vd", type=str, default=None, help='something with json')
    parser.add_argument("--save_json_path_vs", type=str, default=None, help='something with json')   
    parser.add_argument("--output_file_name", type=str, default=None, help='something with json')
    parser.add_argument("--gpt_model", type=str, default='None')
    parser.add_argument("--api_key", type=str, default='None')
    parser.add_argument("--conv-mode", type=str, default="llava_v1")

    args = parser.parse_args()

    print(f'EVAL USING {args.gpt_model} MODEL')

    image_folder=args.image_folder
    model_path=args.model_path
    model_base=args.model_base

    data_vd = []
    data_vs = []
    # saving response to run 3 times 
    if not os.path.isfile(args.output_file_name):
        
        with open(input_file_name) as json_file:
            datas = json.load(json_file)

        datas = generate_answer(datas, model_output_entry)

        # save the response
        with open(args.output_file_name, 'w') as f:
            json.dump(datas, f, indent=4)

    else:
        print(f'loading file from {args.output_file_name}')
        with open(args.output_file_name) as json_file:
            datas = json.load(json_file)

    for data in tqdm(datas):
        if data['category'] == 'VD':
            data_vd.append(data)
        if data['category'] == 'VS':
            data_vs.append(data)
                      
    print('vd: evaluate_by_chatgpt')
    data_vd = evaluate_by_chatgpt(data_vd, model_output_entry, model_correctness_entry, 
                                  load_json=load_json, save_json_path=args.save_json_path_vd,
                                  gpt_model=args.gpt_model, api_key=args.api_key)
    print('vd: check_same_by_chatgpt')
    data_vd = check_same_by_chatgpt(data_vd, model_output_entry, 
                                    load_json=load_json, save_json_path=args.save_json_path_vd, 
                                    gpt_model=args.gpt_model, api_key=args.api_key)
    #time.sleep(60) #
    try:
        print('try, vs: evaluate_by_chatgpt')
        data_vs = evaluate_by_chatgpt(data_vs, model_output_entry, model_correctness_entry, 
                                      load_json=load_json, save_json_path=args.save_json_path_vs, 
                                      gpt_model=args.gpt_model, api_key=args.api_key)
        print('try, vs: check_same_by_chatgpt')
        data_vs = check_same_by_chatgpt(data_vs, model_output_entry, load_json=load_json, 
                                        save_json_path=args.save_json_path_vs, 
                                        gpt_model=args.gpt_model, api_key=args.api_key)
    except:
        time.sleep(60)
        print('except, vs: evaluate_by_chatgpt')
        data_vs = evaluate_by_chatgpt(data_vs, model_output_entry, model_correctness_entry, 
                                      load_json=load_json, save_json_path=args.save_json_path_vs, 
                                      gpt_model=args.gpt_model, api_key=args.api_key)
        print('except, vs: check_same_by_chatgpt')
        data_vs = check_same_by_chatgpt(data_vs, model_output_entry, load_json=load_json, 
                                        save_json_path=args.save_json_path_vs, 
                                        gpt_model=args.gpt_model, api_key=args.api_key)
    print("##### GPT Evaluate #####")

    data_vd = assign_correctness(data_vd, correctness_entry=model_correctness_entry)
    data_vs = assign_correctness(data_vs, correctness_entry=model_correctness_entry)
    data = data_vd + data_vs

    all_data = get_eval_all(data, model_correctness_entry)
    all_vd = get_eval_all(data_vd, model_correctness_entry)
    all_vs = get_eval_all(data_vs, model_correctness_entry)

    table1 = [["per question", "Total"], 
              ["VD", round(100 * all_vd["correct"]/all_vd["total"], 4)], 
              ["VS", round(100 * all_vs["correct"]/all_vs["total"], 4)], 
              ["Overall", round(100 * all_data["correct"]/all_data["total"], 4)]]
    tab1 = PrettyTable(table1[0])
    tab1.add_rows(table1[1:])


    q_acc_gpt = round(100 * all_data["correct"]/all_data["total"], 4)

    all_data = get_eval_pair_all(data, model_correctness_entry)
    easy = get_eval_pair_easy(data)
    hard = get_eval_pair_hard(data)
    all_vd = get_eval_pair_all(data_vd, model_correctness_entry)
    easy_vd = get_eval_pair_easy(data_vd)
    hard_vd = get_eval_pair_hard(data_vd)
    all_vs = get_eval_pair_all(data_vs, model_correctness_entry)
    easy_vs = get_eval_pair_easy(data_vs)
    hard_vs = get_eval_pair_hard(data_vs)
    # question pair level
    table3 = [["per question pair", "Easy", "Hard", "Total"], 
              ["VD", round(100 * easy_vd["correct"]/easy_vd["total"], 4), round(100 * hard_vd["correct"]/hard_vd["total"], 4), round(100 * all_vd["correct"]/all_vd["total"], 4)], 
              ["VS", round(100 * easy_vs["correct"]/easy_vs["total"], 4), round(100 * hard_vs["correct"]/hard_vs["total"], 4), round(100 * all_vs["correct"]/all_vs["total"], 4)], 
              ["Overall", round(100 * easy["correct"]/easy["total"], 4), round(100 * hard["correct"]/hard["total"], 4), round(100 * all_data["correct"]/all_data["total"], 4)]]
    tab3 = PrettyTable(table3[0])
    tab3.add_rows(table3[1:])
    #print(tab3)


    fig_all = get_eval_fig(data)
    fig_vd = get_eval_fig(data_vd)
    fig_vs = get_eval_fig(data_vs)

    # image level 
    table2 = [["per figure", "Correct", "Wrong", "Score"], 
              ["VD", round(100 * fig_vd["correct"]/fig_vd["total"], 4), round(100 * fig_vd["inconsistent"]/fig_vd["total"], 4) + round(100 * fig_vd["wrong"]/fig_vd["total"], 4), round(fig_vd["score"], 4)], 
              ["VS", round(100 * fig_vs["correct"]/fig_vs["total"], 4), round(100 * fig_vs["inconsistent"]/fig_vs["total"], 4) + round(100 * fig_vs["wrong"]/fig_vs["total"], 4), round(fig_vs["score"], 4)], 
              ["Overall", round(100 * fig_all["correct"]/fig_all["total"], 4), round(100 * fig_all["inconsistent"]/fig_all["total"], 4) + round(100 * fig_all["wrong"]/fig_all["total"], 4), round(fig_all["score"], 4)]]
    tab2 = PrettyTable(table2[0])
    tab2.add_rows(table2[1:])

    pair_acc_gpt = round(100 * all_data["correct"]/all_data["total"], 4)
    figure_acc_gpt = round(100 * fig_all["correct"]/fig_all["total"], 4)
    easy_acc_gpt = round(100 * easy["correct"]/easy["total"], 4)
    hard_acc_gpt = round(100 * hard["correct"]/hard["total"], 4)



    print("##### Question Stats #####")
    print("Easy Questions: " + str(easy_vd["total_q"]) + "(Visual Dependent) + " + str(easy_vs["total_q"]) + "(Visual Supplement)")
    print("Hard Questions: " + str(hard_vd["total_q"]) + "(Visual Dependent) + " + str(hard_vs["total_q"]) + "(Visual Supplement)")
    print("Total Questions: " + str(all_data["total_q"]))


    print("##### Figure Stats #####")
    print("Visual Dependent Figures: " + str(fig_vd["total"]))
    print("Visual Supplement Figures: " + str(fig_vs["total"]))
    print("Total Figures: " + str(fig_all["total"]))

    print("##### Leaderboard Stats #####")

    table = [["", "Acc per question pair (qAcc)", "Acc per figure (fAcc)", "Acc per easy question (easy aAcc)", "Acc per hard question (hard aAcc)", "Acc per question (aAcc)"], 
              ["GPT Eval", pair_acc_gpt, figure_acc_gpt, easy_acc_gpt, hard_acc_gpt, q_acc_gpt]]
    leaderboard = PrettyTable(table[0])
    leaderboard.add_rows(table[1:])
    print(leaderboard)


    stats = yes_ratio_stats(data)

    table = [["", "Yes/No Bias (Pct Diff)", "Yes/No Bias (FP Ratio)", "Consistency Test (correct)", "Consistency Test (inconsistent)", "Consistency Test (wrong)", "LH", "VI", "Mixed"], 
              ["GPT Eval", stats["diff"], stats["fp"], round(100 * fig_all["correct"]/fig_all["total"], 4), round(100 * fig_all["inconsistent"]/fig_all["total"], 4), round(100 * fig_all["wrong"]/fig_all["total"], 4), round(100 * all_data["LH_cg"]/(all_data["LH_cg"] + all_data["VI_cg"] + all_data["Mix_cg"]), 4), round(100 * all_data["VI_cg"]/(all_data["LH_cg"] + all_data["VI_cg"] + all_data["Mix_cg"]), 4), round(100 * all_data["Mix_cg"]/(all_data["LH_cg"] + all_data["VI_cg"] + all_data["Mix_cg"]), 4)]]
    test = PrettyTable(table[0])
    test.add_rows(table[1:])
    print(test)

    orig = [i for i in data if int(i["visual_input"]) == 1]

    edit = [i for i in data if int(i["visual_input"]) == 2]

    a = np.unique([i["category"] + "_" + i["subcategory"] + "_" + i["set_id"] + "_" + i["figure_id"] for i in orig])
    b = np.unique([i["category"] + "_" + i["subcategory"] + "_" + i["set_id"] + "_" + i["figure_id"] for i in edit])
    print(len(a))
    print(len(b))



