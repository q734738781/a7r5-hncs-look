"""Fixed region averages, not semantic or learned image features."""
import numpy as np
from scipy.sparse import coo_matrix

def build_pool(groups,n):
    rows=[];cols=[];values=[];weights=[]
    for ids,weight in groups:
        ids=np.asarray(ids,int)
        if not len(ids) or weight<=0:continue
        row=len(weights);rows.extend([row]*len(ids));cols.extend(ids);values.extend([1/len(ids)]*len(ids));weights.append(weight)
    if not weights:raise ValueError('At least one nonempty appearance region is required')
    return coo_matrix((values,(rows,cols)),shape=(len(weights),n)).tocsr(),np.asarray(weights)/sum(weights)
