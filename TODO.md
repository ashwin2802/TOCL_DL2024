Riccardo ToDo: 
- Gather data from Amazon Review dataset, download the files, and write a parser that loads them and generate the dataset (morning)
- Try a simple run, also on the gpu for double-checking that everything works
- Implement some sort of data visualization, or wandb project for storing results

What is required for the experiemnts is the Accuracy Metrix for some sort of post-processing, and with it, every other metric can be recomputed.
- Implement wandb logging of the data, where we log only the final plots, like average accuracy and so on
- Save into json the results of the experiments

What is requried for repeating experiments and let Ashwin continue: 
- set up model loader with .get(model_name), and this returns an implementation of it
- understand whether we want a pre-trained version of it, but this might be too complicated

Repeat experiments from the Cambridge Paper: 
- 2-layer MLP with ReLU activation and sigmoid for binary classification: classification head already has the 10 classes allocated
    ADAM with fixed parameters, and say 5 epochs -> understand which are enough for good performances
- try out some Continual Learning strategies and pick which gives best results: 
    make the plots of average accuracy, average forgetting, backward and forward transfer. Save them as the .json with the scores
- repeat on CIFAR-100, and double check that the Continual Learning strategies are working
    again, plot the metrics for visualizing what is going on
- implement ResNet-18 from the module loader and train on a fixed number of epochs for MNIST and CIFAR-100

# But first, make experiments with the amazon_review_dataset and delete the data from pc
Make run both with fine-tuning existing model, and pretrain a tiny bert from scatch. 
Implement metric for groping at this point is more important. 

# Run ablation
- Right now we have the code for specifying the task groups and the ordering. However, we are stuck at the problem that the actor size is hard-coded,
    and therefore we can have only tasks with compatible actor size. 
- The next step is implementing an actor that has multiple heads, one for each task label. 
-> This seems to be a problem, and we need to test the implementation further. Change a dimension at a time, i.e., first the number of outputs, 
-> then the inputs, because this might have impact on the model. 

Sequence to Sequence: 
- we have the iCaRL for sequence classification being working, and now it is a matter of running some experiments. 
- which are the experiments and the dimensions to even investigate? 

- find the best mixing coefficient for the iCaRL famework

-> Add the cumulative max and cumulative min grouping, and submit the experiments for it. 

# Experiments of MNIST
We compared metrics: 
- cambridge 
- cambridge w/ grad prod

We compared grouping strategies: 
- min/max intergroup heterogeneity

We compared ordering strategies: 
- min/max ordering:
- min/max cumulative ordering

Our baseline is no grouping and no ordering (~80%)
We first assess whether the grouping has an impact on the final average accuracy.

Grouping by the cambridge metric: 
- optimal grouping: ~83%
- optimal grouping & min ordering: 85%
- optimal grouping & min cum ordering: 88%
- optimal grouping & max ordering: 85%
- optimal grouping & max cum ordering: 89.65%

Grouping by the cambrdge metric and gradient product:
- optimal grouping: ~89.6% -> statistical significance
- optimal grouping & min ordering: 87.36%
- optimal grouping & min cum ordering: 85%
- optimal grouping & max ordering: 83%
- optimal grouping & max ordering: 84%

# Repeat experiments on CIFAR-100
We have already the similarity matrix being computed for CIFAR-100 with 20 and 100 classes, so we first implement the 
LP max partitioning solution.


# Study the Correlation
Let H_i \in R^{m, d}, where m is the number of tasks, and d is the input dimension to the classification heads.
We compute H_i after having learn task i by encoding all test images with the feature extractor, and averaging the obtained hidden representation.

Collecting such matrices over ci_iterations and over all experiences, we obtain a list of [ci_iteration, num_tasks, num_tasks, d]. 

The description of the metrics can be made as mathematical as desired. 

-> trajectory
Linear regression plot saved to: ./plots/CIFAR-100/linear_regression_forgetting_vs_trajectory_lengths_plot.png
Linear regression coefficients: [-7.59405529]
Intercept: 420.63385072139044
Correlation Coefficients: {'pearson_corr': -0.6448634710770506, 'pearson_p_value': 2.894433514152444e-05, 'spearman_corr': 0.006162464985994398, 'spearman_p_value': 0.9719729955454363}

-> kl_divergence
Linear regression plot saved to: ./plots/CIFAR-100/linear_regression_forgetting_vs_kl_divergence_plot.png
Linear regression coefficients: [-0.00142392]
Intercept: 0.127542934533363
Correlation Coefficients: {'pearson_corr': -0.5100603618120908, 'pearson_p_value': 0.001747067648841618, 'spearman_corr': -0.19579831932773112, 'spearman_p_value': 0.25963964840942105}

-> variance
Linear regression plot saved to: ./plots/CIFAR-100/linear_regression_forgetting_vs_variance_plot.png
Linear regression coefficients: [-0.07757147]
Intercept: 8.772092142301178
Correlation Coefficients: {'pearson_corr': -0.470176940982578, 'pearson_p_value': 0.003808753376598711, 'spearman_corr': -0.09858429858429861, 'spearman_p_value': 0.5673037310972122}