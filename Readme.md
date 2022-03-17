# On Learning Implicit Protected Attributes

This repository aims to experiment and build upon the recent reseacrh paper: [Reading Race: AI Recognises Patient's Racial Identity In Medical Images](https://arxiv.org/abs/2107.10356) which explores the capacity of deep neural network to classify a patient race from his chest x-ray. The paper provides evidence that such models can indeed guess the race with significant accuracy even if the images where distorted for a certain degree. The finding raises question if such models can implicilty use race information which is a protected attribute when used to classify diseases.

To answer the question this repository follow the following methodology:

1- Train networks to detect diseases and then train on the resulting freezed network to detect race. After that compare the performance to the performance when trained using freezed network pretrained using ImageNet and freezed non trained networks. [Note: Code Done]

2- Train multi head network on desieases and race simulatenously. Then figure the neurons on the logit level that mostly contributes to races detection adn turn them off. Quantify the effect of turning them off on desies detection performance. [Note: Code Draft Done]

To remove any bais in performance we created a balanced data set over race. The code for spliiting is available with different options.

For the code to work the Chexpert small dataset have to be doanload from [chexpert official site](https://stanfordmlgroup.github.io/competitions/chexpert/).

Below is an examperiment snippet:





