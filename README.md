# TOCL_DL2024

This repository contains the code for the Deep Learning HS2024 project **"How to Order your Tasks: Investigating Task Sequencing in Continual Learning"**.

**Note:**
- In the code, the term **"Cambridge similarity"** refers to the curvature-based task similarity.
- The term **"Cambridge similarity with gradient product"** refers to the curvature-based task similarity augmented with the gradient inner product score, as described in the report.

---

## Reproducing Experiments

To reproduce the experiments, create the Conda virtual environment using the `environment.yml` file.

### Similarity Matrix Computation

Compute the similarity matrices for the MLP (on MNIST-10) and ResNet (on CIFAR-100) architectures using the following commands:

```bash
# For MNIST-10
python similarity/MNIST-10_cambridge_similarity.py \
    --config config/MNIST-10/MNIST-10_cambridge_similarity_epochs_10_task_aware_simpleMLP.json

python similarity/MNIST-10_cambridge_similarity_with_grad_product.py \
    --config config/MNIST-10/MNIST-10_cambridge_similarity_with_grad_product_epochs_10_task_aware_simpleMLP.json

# For CIFAR-100
python similarity/CIFAR-100_cambridge_similarity.py \
    --config config/CIFAR-100/CIFAR-100_cambridge_similarity_epochs_10_task_aware_resnet.json

python similarity/CIFAR-100_cambridge_similarity_with_grad_product.py \
    --config config/CIFAR-100/CIFAR-100_cambridge_similarity_with_grad_product_epochs_10_task_aware_resnet.json
```

The matrices will be saved in the `similarity_matrices/` folder with the naming convention:

```
{dataset}_{similarity_metric}_{model_name}_epochs_10.json
```

We provide examples of these matrices in the repository.

---

### Optimal Task Grouping

Based on the similarity matrices, compute the optimal task groups by running the following:

```bash
python similarity/MNIST-10_compute_optimal_grouping.py
python similarity/CIFAR-100_compute_optimal_hierarchical_grouping.py
```

**Note:**
- Update the `path_to_similarity_matrix` variable in both scripts to point to the respective similarity matrix. Example paths are provided in the code.
- Update the `path_to_opt_grouping` variable to include a placeholder `{}` for adapting the output file path with "min" or "max" to distinguish grouping by minimizing or maximizing the similarity, respectively.

---

### Optimal Task Ordering

With the similarity matrices and task groupings, compute the optimal task orderings by running the following:

```bash
python similarity/MNIST-10_compute_optimal_ordering.py
python similarity/CIFAR-100_compute_optimal_hierarchical_ordering.py
```

**Note:**
- Paths should be updated similarly to the task grouping scripts.
- Example initializations are provided, compatible with the task grouping examples.

---

### Running Experiments

To reproduce experiments and run the MLP or ResNet model with a specific task grouping and ordering, use the provided configuration templates. For example:

```bash
python pipeline/CIFAR-10_manual_impl_with_repr \
    --config config/MNIST-10/MNIST-10_classes_per_task_2_epochs_20_random_grouping.json
```

This evaluates the MLP model on MNIST-10 with random task grouping and random task ordering.

To evaluate specific task groupings and orderings:
1. Adapt the template `config/MNIST-10/MNIST-10_classes_per_task_2_epochs_20_cambridge_similarity_with_grad_prod_optimal_min_grouping.json`.
2. Set the `path_to_task_groups` variable appropriately.

**Note:**
- Set `path_to_task_groups` to `null` if no ordering or grouping is desired.
- Set `shuffle_task_groups` to `false` if the task order should remain unchanged.
- Adjust the `similarity_metric`, `grouping`, and `ordering` variables to the most descriptive terms for your setup. Suggestions are provided in the default configurations.

Example results are available in the `/results` folder.

---

### Analytics on Model's Hidden Representations

To analyze the hidden representations and compute metrics such as trajectory length, variance, and KL-divergence scores during training (see the report for precise characterizations):

```bash
python analytics/postprocess_results.py
```

**Note:**
- By default, the script iterates over all `.json` files in the `/results` folder.
- It updates the `.json` files with analytics scores under the `analytics` attribute.
- Rerunning the script multiple times will not affect the results.

---
