# milan-telecom-analysis
These code is a technical analisys of [A multi-source dataset of urban life in the city of Milan and the Province of Trentino](https://www.nature.com/articles/sdata201555) paper.

### Codes
The routines in these repository where:

 - *libs/get_milano.py*: a library build to get the requested data from the dataset.
 - *libs/functions.py*: NMAE function implementation to evaluate the models.
 - *legacy: old code.
 - *database_adapt.py*: code to get the original database and pass to a format of all data from every cell in one respective csv file.
 - *model_building.py*: construction of model for cells.
 - *transport_modelling*: modelling of transport hubs locations and respective cell that covers it.

### legacy/results
On *compiled_results* folder there are the preliminar results. The *.png* images are the plot of data on *.csv* files.
- *1.png*, *2.png*, *3.png*, *4.png*, *5.png* where the results of the neural networks with 1 neighborhood data, 1 and 2 neighborhood, 1, 2 and 3 neighborhood and so on.
- "*Mean MSE.png*" where the compiled results of the MSE over the epochs for each strategy.
- *Boxplot.png* where an overview for every strategy.

On *model_results* folder there is the models generated from the executions.

On *bkp_1.0* there are the same files as listed above, but with a MSE evaluation over the epochs.

On *nonlinear_results* folder there are the preliminar results using feature selection. The files in it follows the same structure as compiled_results folder.

On *misc* there are some test plots.
