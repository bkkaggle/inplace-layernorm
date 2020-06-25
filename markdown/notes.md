# Notes

-   make custom bn and compare speed
-   make `autograd.function` version of bn and relu
-   combine to make activated batchnorm
-   make checkpointing version
-   check memory usage

# Benchmarks

| impl   | speed  |
| ------ | ------ |
| pt     | 88it/s |
| custom | 78it/s |
