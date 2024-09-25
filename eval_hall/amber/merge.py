
import os
import json
import glob
import sys

def merge_discrimintaive_responses(root, dirname):
    def load_jsonl(filename):
        return [json.loads(q) for q in open(os.path.expanduser(filename), "r", encoding='utf-8')]


    if os.path.isfile(os.path.join(root, dirname, 'amber', 'answer_amber_d.jsonl')):
        print('merged file exists')
        return
    
    json_files=glob.glob(os.path.join(root, dirname, '*amber_d*'))

    inference_data = []
    for f in json_files:
        inference_data.extend(load_jsonl(f))


    new_file=os.path.join(root, dirname, 'amber', 'answer_amber_d.jsonl')
    print(new_file)
    with open(new_file, 'w') as f:
        # json.dump(inference_data, f)

        for k in inference_data:

            f.write(json.dumps(k) + "\n")
            f.flush()
        

if __name__=='__main__':

    root='/fs01/home/anonymous/anonymous_ssd004/OUTPUTS/CKDv2'
    # dirname='Va100_05012021133412_ep_1_alpha_15_lr_5e-6_rand_ub_0.8_lb_0.4_adv_ub_1_lb_0.2_ds_full-qa-balanced_ref_anno_llava_mixed_25K_ref_model_7b_loss_mean_hallava-7b-lora'
    dirname=sys.argv[1]

    merge_discrimintaive_responses(root, dirname)
