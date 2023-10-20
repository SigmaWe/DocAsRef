import sys
import re
import os

import json

path = sys.argv[1]
save = sys.argv[2]
with open(path) as r, open(save,'w') as w:
	for line in r.readlines():
		line = json.loads(line)
		base = line['ll_base']
		llhelp = line['ll_help']
		summlen = line['num_summ_tokens']
		doclen = line['num_doc_tokens']
		cr = 1-float(summlen/doclen)
		#line['CompRatio']=cr
		line['infodiffF']=base-(llhelp*cr/(llhelp+cr))
		w.write(json.dumps(line)+'\n')

