'''
Effectiveness Evaluation Content
1. How many action_history entries did self accept?
2. Self's action_space.
3. How many beliefs were received?
4. How many relationships were received?
5. Is the update score within the specified range?

Reasonability Evaluation
1. Generate thought, goal, and action to let GPT judge whether the behavior is consistent.
2. All content spoken by a certain character.
3. All characters.
4. Design specific action history to judge:
    - The updated value of relationship.
    - The updated value of belief.
    - The rationality of guess.
    - The rationality of vote.
    - The rationality of vote except self.
5. For all endings, the differences between guess, vote, and vote except self.
6.
'''
import collections
import json
import os.path

import numpy
from config import TEST_FOLDER
# evaluate_saving_dir = './storage/test_version'
# evaluate_saving_dir = './storage/succession/saving/gpt35_7'
suc_dir = './storage/succession/saving'
# action_history_dir = os.path.join(evaluate_saving_dir, 'action_history')
chn = True
qe = False
n_gram=2


if qe:
    print('='*50)
    print('Quantity Evaluation')
    print('='*50)
    print()

    for evaluate_saving in os.listdir(suc_dir):
        if evaluate_saving == 'initial_version': continue
        evaluate_saving_dir = os.path.join(suc_dir, evaluate_saving)
        if not os.path.isdir(evaluate_saving_dir): continue
        action_history_dir = os.path.join(evaluate_saving_dir, 'action_history')
        line_to_be_evaluated = collections.defaultdict(list)
        for json_file in os.listdir(action_history_dir):
            json_data = open(os.path.join(action_history_dir, json_file), encoding='utf-8')
            for json_line in json_data:
                json_line = json.loads(json_line)
                action_type = json_line['action_type']
                action = json_line['action']
                agent_response = action.split('agent response: ')[-1].split('[SEP]')[0].strip()
                ground_truth = action.split('ground truth: ')[-1].strip()

                if action_type.startswith('### EVALUATION') and agent_response != '[SKIP]':
                    line_to_be_evaluated[action_type].append([agent_response, ground_truth])
        if line_to_be_evaluated:
            print(evaluate_saving)
            [print(j,'mean=',numpy.mean([1 if i[0] == i[1] else 0 for i in line_to_be_evaluated[j]])) for j in line_to_be_evaluated]

print('='*50)
print('Entropy Evaluation')
print('='*50)
print()
for evaluate_saving in os.listdir(suc_dir):
    if evaluate_saving == 'initial_version': continue
    evaluate_saving_dir = os.path.join(suc_dir, evaluate_saving)
    if not os.path.isdir(evaluate_saving_dir): continue
    action_history_dir = os.path.join(evaluate_saving_dir, 'action_history')
    n_gram_dict = collections.defaultdict(int)
    count = 0
    for json_file in os.listdir(action_history_dir):
        json_data = open(os.path.join(action_history_dir, json_file), encoding='utf-8')
        for json_line in json_data:
            json_line = json.loads(json_line)
            action_type = json_line['action_type']
            action = json_line['action']
            if not chn:
                action = action.split(' ')
            if action_type == '### SAY' or action_type.startswith('### SPEECH'):
                for i in range(len(action)-n_gram+1):
                    n_gram_dict[action[i:i+n_gram]] += 1
                    count += 1
    entropy = 0
    for key, item in n_gram_dict.items():
        p = item/count
        entropy -= p * numpy.log(p)
    print('%30s, %d-gram count: %7d, Entropy: %7.5f, All Token: %10d'%(evaluate_saving,n_gram, len(n_gram_dict),entropy, count))






