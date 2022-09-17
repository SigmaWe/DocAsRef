
The result below has the following cofigurations: 
1. `idf` in BERTScore is False/default. 

```
corr_metric  aspect                 approach  model               score_name
pearsonr     CoherenceRating        trad      rouge               rouge1       -0.081
                                                                  rouge2       -0.112
                                                                  rougeL       -0.107
                                                                  rougeLsum    -0.107
                                    new       rouge               rouge1        0.621
                                                                  rouge2        0.643
                                                                  rougeL        0.650
                                                                  rougeLsum     0.650
                                    trad      bertscore           precision    -0.045
                                                                  recall        0.240
                                                                  f1            0.074
                                                                  hashcode        NaN
                                    new       bertscore           precision     0.646
                                                                  recall        0.714
                                                                  f1            0.688
                                                                  hashcode        NaN
                                    trad      bleurt              scores        0.027
                                    new       bleurt              scores        0.612
                                    trad      bertscore-sentence  P             0.120
                                                                  R             0.206
                                                                  F             0.161
                                    new       bertscore-sentence  P             0.660
                                                                  R             0.607
                                                                  F             0.646
             FluencyRating          trad      rouge               rouge1       -0.084
                                                                  rouge2       -0.105
                                                                  rougeL       -0.104
                                                                  rougeLsum    -0.104
                                    new       rouge               rouge1        0.558
                                                                  rouge2        0.590
                                                                  rougeL        0.588
                                                                  rougeLsum     0.588
                                    trad      bertscore           precision    -0.033
                                                                  recall        0.213
                                                                  f1            0.066
                                                                  hashcode        NaN
                                    new       bertscore           precision     0.590
                                                                  recall        0.660
                                                                  f1            0.630
                                                                  hashcode        NaN
                                    trad      bleurt              scores        0.018
                                    new       bleurt              scores        0.588
                                    trad      bertscore-sentence  P             0.099
                                                                  R             0.195
                                                                  F             0.146
                                    new       bertscore-sentence  P             0.625
                                                                  R             0.564
                                                                  F             0.601
             InformativenessRating  trad      rouge               rouge1       -0.050
                                                                  rouge2       -0.091
                                                                  rougeL       -0.085
                                                                  rougeLsum    -0.085
                                    new       rouge               rouge1        0.779
                                                                  rouge2        0.788
                                                                  rougeL        0.788
                                                                  rougeLsum     0.788
                                    trad      bertscore           precision    -0.071
                                                                  recall        0.279
                                                                  f1            0.075
                                                                  hashcode        NaN
                                    new       bertscore           precision     0.684
                                                                  recall        0.803
                                                                  f1            0.749
                                                                  hashcode        NaN
                                    trad      bleurt              scores        0.077
                                    new       bleurt              scores        0.606
                                    trad      bertscore-sentence  P             0.171
                                                                  R             0.293
                                                                  F             0.226
                                    new       bertscore-sentence  P             0.733
                                                                  R             0.733
                                                                  F             0.759
             RelevanceRating        trad      rouge               rouge1       -0.013
                                                                  rouge2       -0.059
                                                                  rougeL       -0.054
                                                                  rougeLsum    -0.054
                                    new       rouge               rouge1        0.709
                                                                  rouge2        0.719
                                                                  rougeL        0.714
                                                                  rougeLsum     0.714
                                    trad      bertscore           precision    -0.021
                                                                  recall        0.278
                                                                  f1            0.105
                                                                  hashcode        NaN
                                    new       bertscore           precision     0.682
                                                                  recall        0.746
                                                                  f1            0.725
                                                                  hashcode        NaN
                                    trad      bleurt              scores        0.103
                                    new       bleurt              scores        0.586
                                    trad      bertscore-sentence  P             0.191
                                                                  R             0.284
                                                                  F             0.235
                                    new       bertscore-sentence  P             0.774
                                                                  R             0.733
                                                                  F             0.771
spearmanr    CoherenceRating        trad      rouge               rouge1        0.071
                                                                  rouge2        0.016
                                                                  rougeL        0.016
                                                                  rougeLsum     0.016
                                    new       rouge               rouge1        0.564
                                                                  rouge2        0.591
                                                                  rougeL        0.591
                                                                  rougeLsum     0.591
                                    trad      bertscore           precision     0.044
                                                                  recall        0.309
                                                                  f1            0.185
                                                                  hashcode        NaN
                                    new       bertscore           precision     0.633
                                                                  recall        0.659
                                                                  f1            0.663
                                                                  hashcode        NaN
                                    trad      bleurt              scores        0.075
                                    new       bleurt              scores        0.596
                                    trad      bertscore-sentence  P             0.175
                                                                  R             0.211
                                                                  F             0.193
                                    new       bertscore-sentence  P             0.616
                                                                  R             0.554
                                                                  F             0.618
             FluencyRating          trad      rouge               rouge1        0.073
                                                                  rouge2        0.037
                                                                  rougeL        0.025
                                                                  rougeLsum     0.025
                                    new       rouge               rouge1        0.476
                                                                  rouge2        0.511
                                                                  rougeL        0.515
                                                                  rougeLsum     0.515
                                    trad      bertscore           precision     0.045
                                                                  recall        0.316
                                                                  f1            0.187
                                                                  hashcode        NaN
                                    new       bertscore           precision     0.591
                                                                  recall        0.590
                                                                  f1            0.618
                                                                  hashcode        NaN
                                    trad      bleurt              scores        0.079
                                    new       bleurt              scores        0.562
                                    trad      bertscore-sentence  P             0.164
                                                                  R             0.223
                                                                  F             0.190
                                    new       bertscore-sentence  P             0.601
                                                                  R             0.509
                                                                  F             0.590
             InformativenessRating  trad      rouge               rouge1        0.105
                                                                  rouge2        0.069
                                                                  rougeL        0.035
                                                                  rougeLsum     0.035
                                    new       rouge               rouge1        0.744
                                                                  rouge2        0.746
                                                                  rougeL        0.746
                                                                  rougeLsum     0.746
                                    trad      bertscore           precision    -0.034
                                                                  recall        0.312
                                                                  f1            0.149
                                                                  hashcode        NaN
                                    new       bertscore           precision     0.611
                                                                  recall        0.750
                                                                  f1            0.689
                                                                  hashcode        NaN
                                    trad      bleurt              scores        0.157
                                    new       bleurt              scores        0.549
                                    trad      bertscore-sentence  P             0.204
                                                                  R             0.272
                                                                  F             0.231
                                    new       bertscore-sentence  P             0.586
                                                                  R             0.644
                                                                  F             0.669
             RelevanceRating        trad      rouge               rouge1        0.128
                                                                  rouge2        0.087
                                                                  rougeL        0.063
                                                                  rougeLsum     0.063
                                    new       rouge               rouge1        0.639
                                                                  rouge2        0.648
                                                                  rougeL        0.641
                                                                  rougeLsum     0.641
                                    trad      bertscore           precision     0.012
                                                                  recall        0.291
                                                                  f1            0.171
                                                                  hashcode        NaN
                                    new       bertscore           precision     0.591
                                                                  recall        0.658
                                                                  f1            0.617
                                                                  hashcode        NaN
                                    trad      bleurt              scores        0.147
                                    new       bleurt              scores        0.507
                                    trad      bertscore-sentence  P             0.194
                                                                  R             0.230
                                                                  F             0.224
                                    new       bertscore-sentence  P             0.584
                                                                  R             0.577
                                                                  F             0.610

```