# Notes

-   make custom bn and compare speed
-   make `autograd.function` version of bn and relu
-   combine to make activated batchnorm
-   make checkpointing version
-   check memory usage

# Benchmarks

| impl              | speed  | memory |
| ----------------- | ------ | ------ |
| pt                | 88it/s | 42mb   |
| custom            | 78it/s | 44mb   |
| autograd.function | 70it/s |
