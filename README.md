# Resnet-DPO

This is project is a proof-of-concept to test out the recent Direct Preference Optimization loss in a simple context. I wanted to better understand how to work with this loss in the simple context of classifying images (mainly on the CIFAR10 dataset). We are using a ResNet fine-tuned on CIFAR10 (with added DropOut) as a baseline model to compare the results obtained from optimizing with DPO.  

Since DPO makes use of two networks $\pi_{\theta}$ and $\pi_{ref}$, we propose to start DPO training process with $\pi_{ref}$ as a ResNet network only fine-tuned for 1 epoch. This allows the reference model to have reasonable performance (~75% accuracy) which is recommended in the DPO article. This makes sense because if the model drifts too much to provide good accuracy, the loss will increase due to the implicit KL penalty between the two models (the ratios in the following equation).

The DPO Loss: 

$$L_{DPO}(\pi_{theta};\pi_{ref}) = - E_{(x, y_w, y_l) \sim D}[\log \sigma (\beta \log \frac{\pi_{\theta}(y_w | x)}{\pi_{ref}(y_w | x)} - \beta \log \frac{\pi_{\theta}(y_l | x)}{\pi_{ref}(y_l | x)})]$$

In the case of CIFAR10, we only have ground truth labels, so there's nothing related to preferences. We modeled the winning/preferred labels ($y_w$) as the ground truth class for each image and the losing/dispreferred labels ($y_l$) as a random class. This allows the model to improve as it would with a cross-entropy training.

There are a few parameters of the training scripts that are available to test out such as:
- do_polyak: The reference model parameters are a moving average towards the parameters of the policy model.
- do_copy: Instead of averaging the models parameters together, we just copy the policy into the reference model at the end of each epoch.

> The implementation of DPO is adapted from the original paper: https://arxiv.org/pdf/2305.18290.pdf  
> Rafailov, R., Sharma, A., Mitchell, E., Manning, C. D., Ermon, S., & Finn, C. (2024). Direct preference optimization: Your language model is secretly a reward model. Advances in Neural Information Processing Systems, 36.