# Towards multi-morphology controllers

This is the code for the paper [Towards Multi-Morphology Controllers with Diversity and Knowledge Distillation](https://dl.acm.org/doi/abs/10.1145/3638529.3654013) ([arXiv](https://arxiv.org/abs/2404.14625))

## A *really* short summary

We use QD to build an archive of robot-controller pairs with distinct morphologies, optimized for the locomotion task. This archive can be distilled into a single controller that can also generalize well to unseen morphologies.

<div align='center'>
<img src="assets/teaser.png"></img>
</div>

## Main components of the code

There are three main components:

1) [QD for domain exploration](src/main.py) -- runs MAP-Elites algorithm to build an archive of solutions.
2) [Dataset creation](src/create_dataset.py) -- takes an archive and creates a dataset by simulating individuals.
3) [Distillation](src/distill.py) -- takes a dataset and trains a network with it in a supervised fashion.

## Watch the behaviors of the robots

With a little bit of explanation as well:

https://github.com/mertan-a/towards-multi-morphology-controllers/assets/34231008/e4cf08af-ed28-47c1-a89a-6736ef25e236

# Citation

```
@inproceedings{10.1145/3638529.3654013,
author = {Mertan, Alican and Cheney, Nick},
title = {Towards Multi-Morphology Controllers with Diversity and Knowledge Distillation},
year = {2024},
isbn = {9798400704949},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3638529.3654013},
doi = {10.1145/3638529.3654013},
abstract = {Finding controllers that perform well across multiple morphologies is an important milestone for large-scale robotics, in line with recent advances via foundation models in other areas of machine learning. However, the challenges of learning a single controller to control multiple morphologies make the 'one robot one task' paradigm dominant in the field. To alleviate these challenges, we present a pipeline that: (1) leverages Quality Diversity algorithms like MAP-Elites to create a dataset of many single-task/single-morphology teacher controllers, then (2) distills those diverse controllers into a single multi-morphology controller that performs well across many different body plans by mimicking the sensory-action patterns of the teacher controllers via supervised learning. The distilled controller scales well with the number of teachers/morphologies and shows emergent properties. It generalizes to unseen morphologies in a zero-shot manner, providing robustness to morphological perturbations and instant damage recovery. Lastly, the distilled controller is also independent of the teacher controllers - we can distill the teacher's knowledge into any controller model, making our approach synergistic with architectural improvements and existing training algorithms for teacher controllers.},
booktitle = {Proceedings of the Genetic and Evolutionary Computation Conference},
pages = {367–376},
numpages = {10},
keywords = {evolutionary robotics, soft robotics, brain-body co-optimization},
location = {Melbourne, VIC, Australia},
series = {GECCO '24}
}
```

