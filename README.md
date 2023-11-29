# segmenting_models

The analysis is part of a project to evaluate the predictive power of electrophysiological models of cardiomyocytes. Cardiomyocyte models are commonly used to generate populations of models that hypothesize the electrical behavior of actual cardiac cells. I am interested in understanding if a single model represents one specific cardiac cell or if a subset of the population of models approximates a single cell.

The model datasets used here represent a population of models that have been trained using an evolutionary algorithm (EA) to single-cell dataset. The EA was independently run multiple times (at least 5x) and the 10% of models with the best EA scores were saved from each EA run. The resulting dataset is pooled. 
