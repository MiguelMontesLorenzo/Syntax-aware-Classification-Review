# Syntax-aware-Classification-Review


## Structure
> data_sst (the folder is downloaded when executing)
> models (created automatically when executing)
  > RecurrentModel *
  > RecursiveModel *
> pretrained
  - skipgram_dep_model_updated.pth *
  - vocab_to_int.json
> runs (created automatically)
> src
  > Embeddings
  > Experiments (scope of negation experiments)
  > NaiveBayesModel
  > RecurrentModel
  > RecursiveModel
  > VecAvgModel
  - data.py
  - pretrained_embeddings.py
  - treebank.py
  - utils.py

* If you wish to train your own models you can pretrain your own embeddings with the scrits in the Embeddings folder. You may also download our pretrained embeddings. The models we have trained can also be downloaded if you dont wish to train your own. This files can be found in the following link:
  https://upcomillas-my.sharepoint.com/:f:/r/personal/202108204_alu_comillas_edu/Documents/PROYECTO_NLP_DEEP?csf=1&web=1&e=6x914N


