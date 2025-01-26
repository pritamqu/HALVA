
import os
import json
import glob
import sys

def merge_discrimintaive_responses(dirname):
    def load_jsonl(filename):
        return [json.loads(q) for q in open(os.path.expanduser(filename), "r", encoding='utf-8')]


    if os.path.isfile(os.path.join(dirname, 'amber', 'answer_amber_d.jsonl')):
        print('merged file exists')
        return
    
    json_files=glob.glob(os.path.join(dirname, 'amber', '*amber_d*'))

    inference_data = []
    for f in json_files:
        inference_data.extend(load_jsonl(f))


    new_file=os.path.join(dirname, 'amber', 'answer_amber_d.jsonl')
    print(new_file)
    with open(new_file, 'w') as f:
        # json.dump(inference_data, f)

        for k in inference_data:

            f.write(json.dumps(k) + "\n")
            f.flush()
        

if __name__=='__main__':

    dirname=sys.argv[1]

    merge_discrimintaive_responses(dirname)
