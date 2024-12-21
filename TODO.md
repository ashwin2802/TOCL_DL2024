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

## Cambdridge Paper Task Similarity
We are running a script that computes the similarity matrix for all 100 tasks. 
We are running a script that computes the similarity matrix for all 100 tasks, and normalizes the scores by the g_k squared L2 norm.

We then use such partitioning for running the continual learning algorithms.