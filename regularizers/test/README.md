# Regularizers

Regularizers calculate data about the weights of a network's layers. We
typically add regularization to the loss so that the network "prefers"
better-behaved weights, for some definition of "better-behaved".

For example, `RegularizerL2` calculates the sum of the L2-norms of the weights
in each layer. We typically add this to the loss of the network so that the
network penalizes weights which have a high L2-norm. That is, `RegularizerL2`
makes the network learn to distribute its knowledge throughout the weights
instead of localizing knowledge in any single point.
