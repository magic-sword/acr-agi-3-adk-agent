"""Reaggregate saved weights: isolate chat-template tokens and phrase starts."""
import json,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.benchmark_recognition import save_json
from scripts.benchmark_explanation_xai import TARGET,LAYERS

out=Path(sys.argv[1])
for case in TARGET:
    meta=json.loads((out/f'{case}-attention-meta.json').read_text());trace=json.loads((out/f'{case}-trace.json').read_text());raw=np.load(out/f'{case}-attention.npz')
    labels=meta['labels'][:];tokens=meta['token_strings']
    for i,t in enumerate(tokens[:meta['prompt_tokens']]):
        if t in ['<|im_start|>','<|im_end|>','<|vision_start|>','<|vision_end|>']:labels[i]='chat_template'
        if i and tokens[i-1]=='<|im_start|>':labels[i]='chat_template'
    # These two additional readouts use already-recorded queries, not new model runs.
    for field in ['before_position','after_position']:
        meta['groups'][field+'_start']=[meta['groups'][field][0]]
        index=meta['prompt_tokens']+meta['groups'][field][0]
        trace['scores'][field+'_start']=dict(text=tokens[index].replace('Ġ',' ').replace('Ċ','\n'))
    lookup={i:k for k,i in enumerate(meta['selected_answer_token_indices'])};meta['labels']=labels;trace['attention']=[];trace['image_maps']={}
    vis=raw['visual_indices'];visual_set=set(vis.tolist());label_array=np.array(labels)
    for field,ids in meta['groups'].items():
        trace['image_maps'][field]={};indices=[lookup[i] for i in ids]
        for layer in LAYERS:
            weights=raw[str(layer)];avg=weights[:,indices,:].mean(axis=(0,1));assert np.isclose(avg.sum(),1,atol=1e-6)
            mass={label:float(avg[label_array==label].sum()) for label in sorted(set(labels))}
            top=sorted([i for i in range(len(tokens)) if i not in visual_set],key=lambda i:-avg[i])[:12]
            trace['attention'].append(dict(field=field,layer=layer+1,mass=mass,top_text=[dict(index=i,weight=float(avg[i]),group=labels[i],token=tokens[i].replace('Ġ',' ').replace('Ċ','\n'),context=''.join(tokens[max(0,i-3):i+4]).replace('Ġ',' ').replace('Ċ','\n')) for i in top]))
            trace['image_maps'][field][str(layer+1)]=avg[vis].reshape(3,26,20).tolist()
    save_json(out/f'{case}-trace-refined.json',trace);save_json(out/f'{case}-attention-groups.json',meta)
print('Reaggregated saved weights; no model execution')
