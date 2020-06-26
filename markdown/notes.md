# Notes

# Todo

-   make `autograd.function` version of bn and relu
-   combine to make activated batchnorm
-   make checkpointing version

# Done

-   make custom bn and compare speed
-   check memory usage

# Benchmarks

| impl              | speed  | memory |
| ----------------- | ------ | ------ |
| pt                | 88it/s | 9.5%   |
| custom            | 78it/s | 7.5%   |
| autograd.function | 70it/s | 10.7%  |
