# Historical LaplaceKV tools

`validate_laplace_kv.py` belongs to the earlier cache-evaluation CLI. It invokes
flags removed from the current application, including `--eval-file` and
`--laplace-stream`. Use it only with the historical runtime revision associated
with the result being reproduced. Its `--self-test` checks the runner itself;
it does not establish compatibility with the current Laplace executable.

For current measurements, use the [benchmark guide](../../../docs/benchmarks.md).
