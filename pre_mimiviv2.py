import os
import json
import random
import argparse

if __name__ == '__main__':
    random.seed(42)

    parser = argparse.ArgumentParser()
    parser.add_argument('--renew_indices', default=True, type=bool)

    args = parser.parse_args()
    renew_indices = args.renew_indices
    print('renew_indices', renew_indices)
    if renew_indices:
        for json_name in ['mimic-iv-discharge', 'mimic-iv-radiology']:

            target_dict = None
            with open('./dataset/' + json_name + '.json', 'r') as f_read:
                target_dict = json.load(f_read)

            total_size = len(target_dict['data'])
            rand_order = list(range(total_size))
            random.shuffle(rand_order)

            eval_test_amount = round(total_size * 0.1)
            train_list, eval_list, test_list = rand_order[:-2 * eval_test_amount],  rand_order[-2 * eval_test_amount:-1 * eval_test_amount], rand_order[-1 * eval_test_amount:]
            print(len(train_list), len(eval_list), len(test_list))
            
            idx_dict = {
                'train': sorted(train_list),
                'eval': sorted(eval_list),
                'test': sorted(test_list),
            }
            with open(os.path.join('./dataset/' + json_name + '_indices.json'), 'w', encoding='utf-8') as write_f:
                write_f.write(json.dumps(idx_dict))

    for json_name in ['mimic-iv-discharge', 'mimic-iv-radiology']:
        idx_dict = None
        target_dict = None
        with open(os.path.join('./dataset/' + json_name + '_indices.json'), 'r', encoding='utf-8') as f_read:
            idx_dict = json.load(f_read)
        
        with open('./dataset/' + json_name + '.json', 'r') as f_read:
            target_dict = json.load(f_read)

        target_dict = target_dict['data']

        write_json = dict()
        for key, value in idx_dict.items():
            
            write_json[key] = [target_dict[num] for num in value]
            print(json_name, key, len(write_json[key]))

        with open(os.path.join('./dataset/' + json_name + '_split.json'), 'w', encoding='utf-8') as write_f:
            write_f.write(json.dumps(write_json))

    print('====== Done ======')