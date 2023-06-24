import argparse
import json
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    OpenAIGPTLMHeadModel,
    OpenAIGPTTokenizer,
    XLNetLMHeadModel,
    XLNetTokenizer,
    TransfoXLLMHeadModel,
    TransfoXLTokenizer,
    ReformerModelWithLMHead,
    ReformerTokenizer,
    XLMWithLMHeadModel,
    XLMTokenizer,
)
import numpy as np
from nltk import sent_tokenize
from nltk import word_tokenize

import scipy
from scipy import spatial

def get_model(name, size):
    if name == 'gpt2':
        t = GPT2Tokenizer.from_pretrained('gpt2')
        if size == 'base':
            g = GPT2LMHeadModel.from_pretrained('gpt2')
        else:
            g = GPT2LMHeadModel.from_pretrained(f'gpt2-{size}')
        eos = g.config.eos_token_id # '<|endoftext|>'
        max_input = 1024
    elif name == 'gpt1':
        t = OpenAIGPTTokenizer.from_pretrained('openai-gpt')
        g = OpenAIGPTLMHeadModel.from_pretrained('openai-gpt')
        eos = 1 # .
        max_input = 512
    elif name == 'xlnet':
        t = XLNetTokenizer.from_pretrained(f'xlnet-{size}-cased')
        g = XLNetLMHeadModel.from_pretrained(f'xlnet-{size}-cased')
        eos = g.config.eos_token_id # </s>
        max_input = 1024
    elif name == 'transformerxl':
        t = TransfoXLTokenizer.from_pretrained('transfo-xl-wt103')
        g = TransfoXLLMHeadModel.from_pretrained('transfo-xl-wt103')
        eos = g.config.eos_token_id # <eos>
        max_input = 1024
    elif name == 'reformer':
        t = ReformerTokenizer.from_pretrained('google/reformer-crime-and-punishment')
        g = ReformerModelWithLMHead.from_pretrained('google/reformer-crime-and-punishment')
        eos = g.config.eos_token_id # (empty string)
        max_input = 1024
    elif name == 'xlm':
        t = XLMTokenizer.from_pretrained('xlm-clm-ende-1024')
        g = XLMWithLMHeadModel.from_pretrained('xlm-clm-ende-1024')
        g.config.lang_id = g.config.lang2id['en']
        eos = 4 # </s>
        max_input = 1024

    g = g.to('cuda')
    g.eval()
    return g, t, eos, max_input

class Shannon:
    def __init__(
        self,
        verbose=False,
        language_model='gpt2',
        model_size='base',
        num_upstream=0,
        return_token_lls=False,
    ):
        self.verbose = verbose
        self.language_model = language_model
        self.num_upstream = num_upstream
        self.return_token_lls = return_token_lls
        self.g, self.t, self.eos, self.max_input = get_model(language_model, model_size)

    def measure(self, doc_tokens, prompt):
        eos = torch.LongTensor([self.eos]).to('cuda')
        if prompt is None or (prompt.dim() == 1 and len(prompt) == 0):
            prompt = torch.LongTensor([]).to('cuda')

        token_lls = []
        success = []
        past = None
        distribution = []
        for i, token in enumerate(doc_tokens):
            upstream = doc_tokens[:i]
            if len(upstream) + len(prompt) + 1 > self.max_input:
                upstream = upstream[-(self.max_input - 1 - len(prompt)):]
                if past is not None:
                    past = [t[:, :, :, 1:, :] for t in past]

            prefix = torch.cat([eos, prompt, upstream]).unsqueeze(0)
            inputs = self.g.prepare_inputs_for_generation(prefix, past=past, use_cache=True, use_mems=True)
            #print(inputs)
            with torch.no_grad():
                out = self.g(**inputs)
                #print(out)

            if self.language_model in 'gpt2':
                #print(type(out))
                #logits, past = out
                logits = out.logits
            elif self.language_model in ['gpt1', 'reformer']:
                logits, = out
                logits = logits[0, -1, :]
            elif self.language_model == 'xlnet':
                logits, = out
            elif self.language_model == 'transformerxl':
                logits, past = out
                logits = logits[0, -1, :]
            elif self.language_model == 'xlm':
                logits, = out
                logits = logits[0, -1, :]
            #print(logits)
            #print(F.softmax(logits, dim=-1))
            #exit()
            probs = F.softmax(logits, dim=-1).view(-1)
            prob = probs[token].item()

            #print('prob', prob)
            log_prob = np.log(prob)#revise
            #print(log_prob)
            #exit()
            distribution.append(log_prob)
            '''
            token_lls.append(log_prob)
            success.append(int(token == probs.argmax()))

            true_token = self.t.decode([token])
            try:
                pred_token = self.t.decode([probs.argmax()])
            except:
                pred_token = None
            info = -log_prob / np.log(2)
            self.log(f'{true_token},{info}')

        #print(success)
        #exit() 

        return token_lls, success, distribution
            '''
        return distribution

    def go(self, doc, summ, measure_t=False, measure_summ=False):
        sents = sent_tokenize(doc)
        encode_args = {'return_tensors': 'pt'}
        if self.language_model == 'transformerxl':
            encode_args['add_space_before_punct_symbol'] = True
        if self.language_model in ['xlnet', 'xlm']:
            encode_args['add_special_tokens'] = False

        sents_tokens = [self.t.encode(sent, **encode_args).to('cuda').view(-1) for sent in sents]
        summ_tokens = self.t.encode(summ, **encode_args).to('cuda').view(-1)
        sents_tokens = [sent_tokens[:self.max_input - 1 - len(summ_tokens)] for sent_tokens in sents_tokens]
        doc_tokens = torch.cat(sents_tokens, dim=-1)

        if measure_t:
            ll, tries, success = 0, 0, []
            for sent_tokens in sents_tokens:
                sent_ll, sent_tries, sent_success= self.measure(sent_tokens, sent_tokens)
                ll += sent_ll
                tries += sent_tries
                success += sent_success
            return ll, tries, success

        elif measure_summ:
            summ_ll, summ_success = self.measure(summ_tokens, None)
            return summ_ll

        else:
            token_dt_base, token_dt_help, token_dt_full = [], [], []
            S = [[0, 0], [0, 0]]
            for sent_idx in range(len(sents_tokens)):
                sent_tokens = sents_tokens[sent_idx]
                upstream_tensors = sents_tokens[sent_idx-self.num_upstream:sent_idx]
                if len(upstream_tensors) > 0:
                    upstream_context = torch.cat(upstream_tensors)
                else:
                    upstream_context = torch.LongTensor([]).cuda()

                base_prompt = upstream_context
                help_prompt = torch.cat([summ_tokens, upstream_context])
                full_prompt = torch.cat([upstream_context, sent_tokens, upstream_context])

                base_dt_dist = self.measure(sent_tokens, base_prompt)
                help_dt_dist = self.measure(sent_tokens, help_prompt)
                full_dt_dist = self.measure(sent_tokens, full_prompt)

                token_dt_base += base_dt_dist
                token_dt_help += help_dt_dist
                token_dt_full += full_dt_dist

        return token_dt_base, token_dt_help, token_dt_full, len(doc_tokens), len(summ_tokens)
        '''
                results = dict()
                results['dt_base'] = base_dt_dist
                results['dt_help'] = help_dt_dist
                results['dt_full'] = full_dt_dist
                results['doclen'] = len(doc_tokens)
                results['summlen'] = len(summ_tokens)
        return results
        '''

    def log(self, s=None):
        if self.verbose:
            print(s)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--simple', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--measure_t', action='store_true')
    parser.add_argument('--measure_summ', action='store_true')
    parser.add_argument('--input_file', type=str)
    parser.add_argument('--save', type=str)
    parser.add_argument('--eval', type=str, choices=['6', '47', '23'], default=None)
    parser.add_argument('--system', type=str, default=None)
    parser.add_argument('--lm', type=str, default='gpt2')
    parser.add_argument('--model_size', type=str, default='base')
    parser.add_argument('--num_upstream', type=int, default=0)
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--maxtoken', type=int, default=300)
    args = parser.parse_args()

    s = Shannon(args.verbose, args.lm, args.model_size, args.num_upstream)

    if args.simple:
        #doc = 'Jack drove his minivan to the bazaar to purchase milk and honey for his large family'
        #summ = 'Jack bought milk and honey from the bazaar'
        doc = '( CNN ) A North Pacific gray whale has earned a spot in the record books after completing the longest migration of a mammal ever recorded . The whale , named Var v ara , sw am nearly 14 , 000 miles ( 22 , 500 ),   to a release from Oregon State University , whose helped conduct the whale - tracking study . Var v ara , which is Russian for ” Bar bara ,” left her primary feeding ground off Russia ’s S akh alin Island to cross the Pacific Ocean and down the West Coast of the United States to B aja , Mexico . Var v ara ’s journey surpassed a record listed on the Guinness Worlds Records website . It said the previous record was setbyahumpbackwhalethatswamamere10,190-mileround trip'
        summ1 = 'Varvara the gray whale traveled from Russia to Mexico, a swim of record breaking length.'
        summ2 = 'The round humpback has told CNN mammals that Baja was a previous Pacific website for ”Guinness.”'
        print(doc)
        print(summ1)
        print('Summ1:')
        results1 = s.go(doc, summ1, measure_t=args.measure_t, measure_summ=args.measure_summ)
        results2 = s.go(doc, summ2, measure_t=args.measure_t, measure_summ=args.measure_summ)
        x = results1['dt_help']
        y = results2['dt_base'] 
        c = 1 - spatial.distance.cosine(x, y)
        print('CosineSim', c)
        p = scipy.stats.pearsonr(x, y)    # Pearson's r
        print('Pearson',p)
        s = scipy.stats.spearmanr(x, y)   # Spearman's rho
        print('Spearman',s)
        tau = scipy.stats.kendalltau(x, y)  # Ke
        print('Kendalltau',tau)
        print(summ2)
        print('Summ2:')
        x = results2['dt_help']
        y = results2['dt_base'] 
        c = 1 - spatial.distance.cosine(x, y)
        print('CosineSim', c)
        p = scipy.stats.pearsonr(x, y)    # Pearson's r
        print('Pearson',p)
        ss = scipy.stats.spearmanr(x, y)   # Spearman's rho
        print('Spearman',ss)
        tau = scipy.stats.kendalltau(x, y)  # Ke
        print('Kendalltau',tau)
        '''
        #print(x)
        #print(y)
        delt = np.array(y)-np.array(x) 
        zeros = np.zeros(len(delt)) 
        
        c = 1 - spatial.distance.cosine(delt, zeros)
        print('CosineSim', c)
        p = scipy.stats.pearsonr(delt, zeros)    # Pearson's r
        print('Pearson',p)
        s = scipy.stats.spearmanr(delt, zeros)   # Spearman's rho
        print('Spearman',s)
        tau = scipy.stats.kendalltau(delt, zeros)  # Ke
        print('Kendalltau',tau)
        '''

    else:
        with open(args.input_file) as reader:
            if args.input_file.endswith('.jsonl'):
                data = [json.loads(line) for line in reader]
            else:
                data = json.load(reader)

        selection = data[args.start:]
        if args.eval is not None:
            selection = [record for record in data if record['eval'] in args.eval]
        if args.system is not None:
            selection = [record for record in selection if record['model_id'] == args.system]

        with open(args.save,'w') as w:
          for record in tqdm(selection):
            #print(record)
            #exit()
            if args.measure_t or args.measure_summ:
                ll = s.go(
                    record['text'], record['decoded'],
                    measure_summ=args.measure_summ, measure_t=args.measure_t
                )
                '''
                print(json.dumps({
                    'doc_id': record['id'],
                    'system': record['model_id'],
                    'll_summ': ll,
                }))
                '''
                w.write(json.dumps({
                    'doc_id': record['id'],
                    'system': record['model_id'],
                    'll_summ': ll,
                })+'\n')

            else:
                dt_base, dt_help, dt_full, num_doc_tokens, num_summ_tokens = s.go(
                    ' '.join(word_tokenize(record['text'])[:args.maxtoken]), record['decoded']
                ) 
                cr = 1-float(num_summ_tokens/num_doc_tokens)
                x = dt_base
                y = dt_help
                p = scipy.stats.pearsonr(x, y)[0]    # Pearson's r
                ss = scipy.stats.spearmanr(x, y)[0]   # Spearman's rho
                tau = scipy.stats.kendalltau(x, y)[0]  # Ke
                w.write(json.dumps({
                    'doc_id': record['id'],
                    'system': record['model_id'],
                    #'dt_base': dt_base,
                    #'dt_help': dt_help,
                    #'dt_full': dt_full,
                    'num_doc_tokens': num_doc_tokens,
                    'num_summ_tokens': num_summ_tokens,
                    'CompRatio': cr,
					'PearsonF': (p+1)*cr/((p+1)/2+cr),
					'SpearmanF': (ss+1)*cr/((ss+1)/2+cr),
					'KendallF': (tau+1)*cr/((tau+1)/2+cr),
                })+'\n')
