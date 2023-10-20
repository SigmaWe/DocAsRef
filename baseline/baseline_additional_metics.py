# from nltk.translate import meteor_score
import evaluate
# from REUSE_main.mChunker import getChunkBasedScore
from baseline.software.SDC_len import *
from baseline.bart_score import BARTScorer

# class meteor_score_formatted():
    
#     def __init__(self):
#         self.model = meteor_score.single_meteor_score
    
#     # The ALPHA, BETA, GAMMA values are default values based on hugging face's docs
#     # https://huggingface.co/spaces/evaluate-metric/meteor/blob/main/meteor.py
#     def compute(self,predictions, references):
#         result = {'meteor_score':[]}
#         for pred, ref in zip(predictions, references):
#             result['meteor_score'].append(self.model(ref, pred, alpha=0.9, beta=3, gamma=0.5))
#         return result


class meteor_score_formatted():
    
    def __init__(self):
        self.model = evaluate.load('meteor').compute
    
    # The ALPHA, BETA, GAMMA values are default values based on hugging face's docs
    # https://huggingface.co/spaces/evaluate-metric/meteor/blob/main/meteor.py
    def compute(self,predictions, references):
        result = {'meteor_score':[]}
        for pred, ref in zip(predictions, references):
            result['meteor_score'].append(self.model(predictions=[pred], references=[ref])['meteor'])
        return result


class sacrebleu_score_formatted():
    
    def __init__(self):
        self.model = evaluate.load("sacrebleu")
    
    #input is [str,str,str]
    #hard coded the score
    def compute(self, predictions, references):
        result = {'bleu_score':[]}
        for pred, ref in zip(predictions, references):
            result['bleu_score'].append(self.model.compute(predictions=[pred], references=[ref])['score'])
        return result


class bleu_score_formatted():
    
    def __init__(self):
        self.model = evaluate.load("bleu")
    
    #input is [str,str,str]
    #hard coded the score
    def compute(self, predictions, references):
        result = {'bleu_score':[]}
        for pred, ref in zip(predictions, references):
            result['bleu_score'].append(self.model.compute(predictions=[pred], references=[ref])['bleu'])
        return result


# REUSE_Score
# https://aclanthology.org/2022.wmt-1.50/
# https://github.com/AnanyaCoder/WMT22Submission_REUSE
#
##################################
# How to run this metric
# 1. Get code from https://github.com/AnanyaCoder/WMT22Submission_REUSE
# 2. Download and extract Chunker Model at https://iiitaphyd-my.sharepoint.com/:u:/g/personal/ananya_mukherjee_research_iiit_ac_in/Ede2Yu2U9ZBHsD2Yu1PRiZ4BkuDx2GED3E-gXBCaGMlN1Q?e=DvzH8k
# 3. Place the folder in the WMT22Submission_REUSE folder
# 4. You may have to change the variable path_to_Chunker_Model on line 227 to a absolute path for your machine
##################################
# 
# The name getChunkBaseScore is very confusing 
# because in paper its avg of chunk level and sent level
# a close look at this function it computes the chunk level 
# p,r,s = evaluateSentencePair(list1[i],list2[i]) <--------------   function returns precision,
#                                                                   recall, and score as indicated 
#                                                                   on paper section 2.1
#
# 
# 
# 
# Lb_sen_score = getSentenceEmbedding(sentEmbeddingModel,[list1[i],list2[i]])  <-----   This returns the LaBSE sentence score
#                                                                                       the sentEmbeddingModel is defined at the
#                                                                                       bottom of the code, outside of all functions
class REUSE_score():
    
    def __init__(self):
        # self.model = getChunkBasedScore
        pass
        
    def compute(self,predictions, references):
        result = {'REUSE_score':self.model(references, predictions)}
        return result


# SDC*_Score
# https://aclanthology.org/2022.naacl-main.153.pdf
# https://aclanthology.org/attachments/2022.naacl-main.153.software.zip
#
##################################
# How to run this metric
# 1. Get code from https://aclanthology.org/attachments/2022.naacl-main.153.software.zip
# 2. Unzip to EvalBase Folder
# 3. Note: The Code is hardcoded for pytorch gpu ver. if cpu, change all 'cuda' to 'cpu'
##################################
# I only used pearson beacuse the paper said they used pearson's tau for CORR function
class SDC_Star():
    
    def __init__(self):
        self.model = Shannon().go
    
    
    def compute(self,predictions, references):
        result = {'sdc*_score':[]}
        for pred,ref in zip(predictions, references):
            
            dt_base, dt_help, dt_full, num_doc_tokens, num_summ_tokens = self.model(ref,pred)
            cr = 1-float(num_summ_tokens/num_doc_tokens)
            
            x = dt_base
            y = dt_help
            
            p = scipy.stats.pearsonr(x, y)[0]    # Pearson's r
            ss = scipy.stats.spearmanr(x, y)[0]   # Spearman's rho
            tau = scipy.stats.kendalltau(x, y)[0]  # Ke
            
            pearson = (p+1)*cr/((p+1)/2+cr) # This is the formula they wrote for the code in sdc_len.py
            
            result['sdc*_score'].append(pearson)
        return result
    



# BART_Score (cnndm ver)
# https://arxiv.org/abs/2106.11520
# https://github.com/neulab/BARTScore
#
##################################
# How to run this metric
# 1. No need Setup, Just use it, the models are handled by huggingface
###################################
# This is hardcoded for gpu
# The function got the ability to handle List[str, str, str] data, but when actually running it with
# EvalBase it outputs error, therefore, compute single line and append to dict
class BART_Score_Eval():
    
    def __init__(self):
        self.model = BARTScorer(device='cuda', checkpoint='facebook/bart-large-cnn')
            
            
    def compute(self, predictions, references):
        result = {'Bart_Score':[]}
        for pred, ref in zip(predictions, references):
            result['Bart_Score'].append(self.model.score([ref], [pred], batch_size=4)[0])
        return result