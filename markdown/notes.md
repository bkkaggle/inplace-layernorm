# Notes

-   bn backprop: https://kratzert.github.io/2016/02/12/understanding-the-gradient-flow-through-the-batch-normalization-layer.html
-   bn backprop: http://cthorey.github.io./backpropagation/
-   autograd.function: https://pytorch.org/docs/stable/notes/extending.html
-   layernorm: https://qywu.github.io/2019/05/22/explore-gradient-checkpointing.html
-   inplace abn: https://github.com/mapillary/inplace_abn
-   bn mean over axis: https://forums.fast.ai/t/batchnormalization-axis-1-when-used-on-convolutional-layers/214/12

# Todo

-   use resnet 50
-   make checkpointing version
    -   checkpointing version isn't saving any memory
-   make inplace abn

-   invertible autograd.function GeLU
-   inplace layer norm

# Done

-   make custom bn and compare speed
-   check memory usage
-   make `autograd.function` relu
-   combine to make activated batchnorm
-   prob can make autograd.fn bn more efficient by removing something from the cache
-   autograd.fn version implemented properly is almost equivalent to pt version
-   the reason why the checkpointed inplace abn is more memory optimized is because the outputs of the bn aren't cached
-   resnet identity blocks have bn after conv but relu after add.
    -   inplace abn does the same, setting activation to identity for these two modules
        -   https://github.com/mapillary/inplace_abn/blob/master/scripts/modules/residual.py#L67
    -   wideresnet might not have this problem
        -   looks like it doesn't, but will stick with resnet unless something goes wrong
