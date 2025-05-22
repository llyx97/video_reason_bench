import os, json, argparse, re
import os.path as osp
import numpy as np

state_size_map = {3: 1, 4: 2, 1: 1, 2: 2}

def load_json(json_file):
    with open(json_file, 'r') as f:
        datas = json.load(f)
    return datas

def save_json(datas, json_file):
    with open(json_file, 'w') as f:
        datas = json.dump(datas, f, indent=4)

def show_result(results):
    summary = {'overall': [0, 0], \
                'num_op_5_9': [0,0], 'num_op_10_14': [0, 0], \
                'state_1': [0, 0], 'state_2': [0, 0], \
                'order_operation': [0, 0], 'counting_operation': [0, 0], 'infer_state': [0, 0], 'comparison_state': [0, 0], 'prediction_state': [0, 0], 'prediction_operation': [0, 0], \
                'candidates_token_count': 0}

    for id, r in results.items():
        summary['overall'][0] += int(r['rating'])
        summary['overall'][1] += 1
        summary['candidates_token_count'] += r['token_count']['completion_tokens']
            
        state_key = f"state_{state_size_map[int(r['num_state'])]}"
        assert state_key in summary
        if state_key in summary:
            summary[state_key][0] += int(r['rating'])
            summary[state_key][1] += 1

        if int(r['num_operation'])>=5 and int(r['num_operation'])<10:
            num_op_key = 'num_op_5_9'
        elif int(r['num_operation'])>=10 and int(r['num_operation'])<15:
            num_op_key = 'num_op_10_14'
        if num_op_key in summary:
            summary[num_op_key][0] += int(r['rating'])
            summary[num_op_key][1] += 1

        dim = "infer_state" if r['dim'] in ["order_state", "counting_state"] else r['dim']
        assert dim in summary
        summary[dim][0] += int(r['rating'])
        summary[dim][1] += 1

    for key in summary:
        if key.endswith('token_count'):
            print(key, f"{summary[key]/len(results)}")
        else:
            print(key, summary[key], f"{100*summary[key][0]/summary[key][1]:.1f}" if summary[key][1]>0 else 0.0)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()     
    parser.add_argument('--result_path', default="predictions")   
    parser.add_argument('--result_prefix', default=None) 
    parser.add_argument('--overwrite_merge_result', action='store_true')   
    args = parser.parse_args()

    print(f"Model: {osp.basename(args.result_path)}\nConfig: {args.result_prefix}\n")
    merged_file = os.path.join(args.result_path, '_'.join(args.result_prefix.split('_')[:-1])+'.json')
    if os.path.exists(merged_file) and not args.overwrite_merge_result:
        merged_results = load_json(merged_file)
    else:
        merged_results = {}
        result_files = [f"{args.result_path}/{rf}" for rf in os.listdir(args.result_path) if rf.startswith(args.result_prefix)]
        for rf in result_files:
            results = load_json(rf)
            merged_results.update(results)
            # os.remove(rf)
        save_json(merged_results, merged_file)
    
    show_result(merged_results)