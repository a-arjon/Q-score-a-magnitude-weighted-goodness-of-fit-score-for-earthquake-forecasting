# Q-score-a-magnitude-weighted-goodness-of-fit-score-for-earthquake-forecasting
Code for the results in the paper with the same name.

**Thank you to Rafał Wrona, independent researcher from Kraków, Poland for finding the mistakes in the Q-score calculation code.**

The Q and L-Score Evaluation (updated).ipynb notebook contains updated Q-Scores. For these new Q-scores some of the orders changed from the published paper, but the conclusions on it still stand. STEP is still one of the better performing models along with the ETAS_DROneDayMd models (and base ETAS), but throughout the years its background model is better performing (ETAS_DROneDayPPEMd). Only in two of the years did models have a Q-score above 1 though. This may be due to most models developed right now being ETAS variants since they forecast 4-5 magnitude events pretty well; as seen by the L-score and in the Q-score when the highest magnitude events were magnitude 5. However, it is much more important to have models forecast more hazardous events (range 6 and up magnitude) like in 2012 and 2014.


<img width="330" height="250" alt="models_2012" src="https://github.com/user-attachments/assets/ba452f1e-946e-4bfd-a863-9aa9430db9c4" />
<img width="330" height="250" alt="models_2014" src="https://github.com/user-attachments/assets/f773a304-0f40-40cc-bcf4-6b6a50666228" />
<img width="330" height="250" alt="models_2017" src="https://github.com/user-attachments/assets/204b0dd6-8e49-45dc-a65a-2daab86b5494" />

[View the full results](Project/data/Revised yearlyQandLScores.pdf)
