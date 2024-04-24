# H2O Performance Experiments Implementation

This is an implementation of H2O in [FlexGen](https://github.com/FMInference/FlexGen).

More explanation see Appendix A in [paper]().

## Installation

**Requirements**

- PyTorch >= 1.12

```
pip install -e .
or
pip install --use-feature=in-tree-build .
```

## Example

```
cd flexgen
python flex_opt.py --gpu-batch-size 1 --overlap false --hh-ratio 0.1 --hh-all --model facebook/opt-6.7b
or
python flex_opt.py --gpu-batch-size 1 --overlap false --hh-ratio 0.1 --hh-all --model /home/tangm_lab/cse12112106/weights/opt-1.3b --path /home/tangm_lab/cse12112106/weights/opt-1.3b --local
```

## Run Experiments
See test suite in h2o_flexgen/benchmark/h2o/h2o_suite.py

\* The implementation is abused a little bit, because of efficiency concern. More specifically, for n heavy hitter and n locals, we actually preserve n-1 heavy hitter and n+1 locals after the first iteration.
