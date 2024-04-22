# Syntax-aware-Classification-Review
Traditionally, sentiment analysis has been implemented solely on semantic characteristics. As the Stanford paper \cite{Stanford2013} highlights, this semantic-space is not enough to complete this task successfully. To solve that problem, they have introduced the Sentiment TreeBank dataset (SST), which incorporates the syntactical structure of sentences. This paper will analyse the impact of using syntactical characteristics for sentiment analysis using the SST dataset on different recursive models. Futhermore, the addition of more syntactical information to the models will be studied with the use of embeddings pretrained with dependency parsing. Our paper proves that taking into account this new syntactic information has a positive impact on the accuracy.

## Structure
- data_sst (the folder is downloaded when executing)
- models (created automatically when executing)
  - RecurrentModel (*)
  - RecursiveModel (*)
- pretrained
  - skipgram_dep_model_updated.pth (*)
  - vocab_to_int.json
- runs (created automatically)
- src
  - Embeddings
  - Experiments (scope of negation experiments) (**)
  - NaiveBayesModel
  - RecurrentModel
  - RecursiveModel
  - VecAvgModel
  - data.py
  - pretrained_embeddings.py
  - treebank.py
  - utils.py

(*) If you wish to train your own models you can pretrain your own embeddings with the scrits in the Embeddings folder. You may also download our pretrained embeddings. The models we have trained can also be downloaded if you dont wish to train your own. This files can be found in the following link:
  https://upcomillas-my.sharepoint.com/:f:/r/personal/202108204_alu_comillas_edu/Documents/PROYECTO_NLP_DEEP?csf=1&web=1&e=6x914N

(**) In order to be able to execute the experiments fodler, you will need to download the models from the link, as well.

## Results
Our study compares three distinct models for syntax-aware classification: recurrent, recursive, and RNTN (Recursive Neural Tensor Network), as well as Naive Bayes and VecAvg for benchmarking. We have observed that the models that capture more syntactical information achieve better accuracy. Additionally, we see  there is a significant improvement when we introduce embeddings that incorporate syntactic structure. This increase will be particularly notable in simpler models.

Our study shows that the incorporation of syntactic structure within embeddings demonstrates a clear advantage in performance. It results on a higher accuracy, lower loss, and less training time, as the parameter L with all the embeddings is already trained.

The effectiveness of the models varied across different aspects. Notably, the learning rate played a crucial rule on the model's performance. Higher learning rates were essential for effectively training the models, with lower rates leading to worse results. With lower values, the models start with a really high loss and are unable to converge, due to the small step of the learning rate, so they get stuck in local minima.

Furthermore, recurrent models exhibited a tendency to overfit the training data. However, this overfitting tendency can be partly attributed to the small size of the dataset. On another line, recursive models, while theoretically expected to significantly outperform other architectures  due to their ability to capture syntactic dependencies, faced some challenges during training too. This difficulty can be also attributed to the limited size of the dataset. Thus suggesting that the models may require larger datasets for a better performance.

Despite these challenges, our experiments highlight the significance of incorporating syntactic information into classification models, because, as expected, the models with the higher accuracy where recursive models (specially RNTN), even though they did not meet the benchmark of 85\% marked by Stanford \cite{Stanford2013}. On top of that, the comparison of models trained with and without syntactic embeddings clearly demonstrates the advantages of leveraging syntactic structure, particularly in tasks like sentiment analysis.

Additionally, it's worth noting in the results that, since RNTN already incorporates sufficient syntactic features, additional embeddings might not be necessary for it. However, for models lacking such inherent syntactic information, such as the simple RecNN, pretrained embeddings could significantly improve their performance.

Moreover, it could be mentioned that using pretrained embeddings with simpler models might be a more viable option compared to more complex models like RNTN, particularly considering training time constraints.
