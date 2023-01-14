Dimensions:
* level: token, sentence
* similarity measure:
  * cosine
  * MNLI
    * NN 
    * Entail
    * Entail-Contradictory 
* LM: 
  * mpnet
  * BART (not always)
  * RoBERTa 
  * DeBERTa
* size: large, xlarge
* fine-tune strategy: basic (MLM), MNLI 
* Sentence weighting: 
  * PageRank: Sum, Entropy 
* pseudoreference: 
  * AnyRef
    * Pegasus-CNNDM, Pegasus-Newsroom, Pegasus-large, Pegaus-xsum (no need to run/report)
    * BART-CNNDM
  * top-k
  * top-p 

All experiment names must follow the order of dimension above, e.g., `sentence-cosine-DeBERTa-large-MNLI-AnyRef-Pegaus-cnndm`

# Experiments 
* BERTScore-token 
  * RoBERTa-large, DeBERTa-large-mnli, DeBERTa-large (**OOM**)
  * RoBERTa-large-topk, DeBERTa-large-mnli-topk (**not run**)
  * RoBERTa-large-topp, DeBERTa-large-mnli-topp (**top 4**)
  * RoBERTa-large-anyref-{bart, pegasus-{xsum, cnndm, newsroom, large}}-{default, min, mean, constant-\*}, DeBERTa (**not run OOM**)
  * Why no IDF? Error? (**please double check**)
* BERTScore-sentence 
  * cosine: 
    * Vanilla: mpnet (done), RoBERTa-large (done)
    * Add pseudoreference
      * top-k (done), top-p (done)
      * Anyref (**top 3**)
    * Add sentence weighting 
      * mpnet (done: entropy, sum), RoBERTa-large  (**top 2**)
  * MNLI: 
    * Vanilla
      * MNLI-{Bart, RoBERTa}-large-{NN, EO, EC}: done
      * MNLI-DeBERta-xlarge-{NN, EO, EC}: done
      * MNLI-DeBERta-large-{NN, EO, EC}: **top 1**
    * Add sentence weighting 
      * MNLI-DeBERTa-xlarge-EC-{entropy, sum}: done 
      * MNLI-DeBERTa-large-EC-{entropy, sum}: **top 1** 
    * Add pseudorefence 
      * Nothing yet 

