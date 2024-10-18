import json
import io
import numpy as np

splits = ['val','testA','testB']
rd = ['r1', 'r2', 'r3', 'r4', 'r5','r6','r7']
noun_token = []
ret = {}
count = 0


all = json.load(open('/data/yzh/SimREC-main/data/anns/refcoco/refcoco.json', 'r'))
print(len(all['train']), all['val'][420])
# i = 0
# for split in ['train','val']:#splits:
#     # sp = []
#     count = 0
#     for term in all[split]:
#         count += term['cf_id']
#     #     if term['cf_id'] == 0:
#     #         nterm = {}
#     #         nterm['refs'] = term['refs']
#     #         nterm['atts'] = term['atts']
#     #         sp.append(nterm)
#     # ret[split] = sp
#     # count+=len(all[split])
    
#     print(count/len(all[split]))

# for split in splits:
#     for term in all[split][:2]:
#         query = "INPUT: " + term['refs'][0] + " & " + term['atts'][0]
#         print(query)

# with open('/data/yzh/CREC/data/toanno/crefcoco_pos.json', 'w') as fout:
#     json.dump(ret, fout)
# for split in splits:
#     with open('/data/yzh/CREC/data/toanno/crefcoco_nn.txt', 'a') as fout:
#         fout.write(split+'\n')
