# milan-telecom-analysis
These code are a technical analisys of [A multi-source dataset of urban life in the city of Milan and the Province of Trentino](https://www.nature.com/articles/sdata201555) paper.

### Codes
The routines in these repository where:

*libs/*
 - get_milano.py: a library build to get the requested data from the dataset.
 - functions.py: NMAE (Normalized Mean Absolute Error) and MARE (Mean absolute Relative error) implementations.

*transport_modelling*: contains the code to map the transport hubs in Milan. The sources used was [ATM website](https://www.atm.it/en/ViaggiaConNoi/Pages/SchemaReteMetro.aspx), [Wikipedia list of Milan Metro stations](https://en.wikipedia.org/wiki/List_of_Milan_Metro_stations) and [Google Maps Platform](https://developers.google.com/maps?hl=pt-br).
- transport_locations.py: takes a list of metro, tram and bus stations and, from the Google Maps API, saves the coordinates of the stations.
- transport_locations_mapping.py: take the coordinates of every station and find the equivalent region on Milano Grid.

*database_adapt.py*: this code is used to take the original dataframe, that is in a format "one file per day" to a format "one region per day".

*model_building.py*: constructs the purposed model framework for a selected number of regions.

*compile_results.py*: compile the results from constructed models.

*model_building_ARIMA.py*: constructs ARIMA models for a selected number of regions.

*model_building_HW.py*: constructs Holt-Winters models for a selected number of regions.
