# Computing Systemic Risk Measures

This repository contains code to replicate the numerical experiments of
"Computing Systemic Risk Measures with Graph Neural Networks".

The experiments on the stylized datasets in section 6.1 can be replicated by running
"SimpleExperimentOverfit.py" and "SimpleExperimentGeneralize.py"
These scripts will create log files for each run in an "Experiments" directory that will be created automatically.

For the experiments of section 6.2, 6.3 and 6.4 the data must be generated first.
This can be done by running 
"DataGeneration.py"
which will create the validation+test sets and save them in the "Data" directory.

Allocating a fixed amount of bailout capital for the expectation risk measure (Section 6.2)
or the entropic risk measure (not presented in the paper) can be replicated by running
"ExperimentInnerRisk.py" and "ExperimentInnerRiskBenchmark.py".
These scripts will also create log files for each run in the "Experiments" directory.

Computing the systemic risk (=searching for the minimum capital that secures the network) for 
the expectation risk measure (Section 6.3) or the entropic risk measure (Section 6.4) can be 
replicated by running "ExperimentOuter.py" and "ExperimentOuterBenchmark.py"
These scripts will also create log files for each run in the "Experiments" directory. 

The "ExperimentInner/Outer(Benchmark).py" files are structured as follows:
- in the config file you can indicate the parameters of the experiment (e.g., datasets, NN models,
epochs, learning rates, seeds, batch size). Be careful: all possible combinations of parameters will run.
- If you want to add single experiments without running all combinations you can add them to the "manual_exp_configs".
- You can specify the risk measure by manipulating the "risc_func" method.
- There is the option to parallelize the runs of the parameter combinations.
In order to do so just uncomment the "multiprocessing" paragraph and run it instead of the
"debugging loop" paragraph (which you should comment out then)

Finally, "AnalyzeExpData.py" can be used to analyze the data a run of  "ExperimentInner/Outer(Benchmark).py"
by calculating mean and standard deviation across the seeds for all parameter combinations and exporting the
results to csv.

If you have any suggestions or questions, or would like to obtain the validation+test sets used in the paper,
instead of generating them randomly with the DataGeneration file,
please contact the author at: weber@math.lmu.de



