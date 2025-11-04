# Inferring Temporal Dependencies 

In social systems, many interconnected processes continuously interact and respond to internal dynamics and external events. Characterizing temporal relationships within these systems is important but often intractable due to complex or unknown interaction mechanisms. We explore the *cross-correlogram* as a solution to this problem for social data, exploring functional and empirical models to correct for periodic or bursty behaviour across a single trial. This problem is explored in the manuscript available <here> (link to be added when manuscript available). This repository contains code to reproduce results from the paper, calculate cross-correlograms and assess their significance (/functions) alongside an example script (example.ipynb).

In example.ipynb, you can see how to fit the cross-correlogram using a functional,



or empirically smoothed null model



The script `run_validation.py' was used to generate the synthetic validation results in Table 1 of the manuscript. periodic_bursty_results.ipynb contains code to generate plots relating to error in the inhomogeneous periodic and bursty case. 
