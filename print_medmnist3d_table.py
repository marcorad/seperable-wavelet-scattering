PATH = 'medmnist3d-results/'
import os
import json


with open(PATH + 'results.json') as file:
    import json
    results = json.load(file)
    
order = ['organ', 'nodule', 'fracture', 'adrenal', 'vessel', 'synapse']
s = '\t        '
for o in order:
    s += f'{"|" + o:14}'
print(s)
for q in results.keys():
    for d in results[q].keys():
        l = 2 if '[0.75, 0.75]' in q else 1
        s = f'$l={l}$, $d={d[1] if d[1] != "1" else 12}$ '
        s = f'{s:16}'
        for dset in order:
            r = results[q][d][dset]
            auc = f'{r["auc"]:.3f}'[1:]
            s += f'& {auc} & {r["acc"]*100:.1f} '
        s += '\\\\'
        print(s)
