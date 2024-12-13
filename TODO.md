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

