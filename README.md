# MTP-NT: A Mobile Traffic Predictor Enhanced by Neighboring and Transportation Data

<p align="center">
<img src="images/bing_framework_representation.jpg" alt="MTP-NT by Bing" width="400"/>
</p>

These code are a technical analisys of [A multi-source dataset of urban life in the city of Milan and the Province of Trentino](https://www.nature.com/articles/sdata201555) paper and the development of a predictive model to forecast network traffic. The work was carried out during the master's program at the Federal University of Uberlândia.


## Table of Contents

- [Introduction](#introduction)
- [Getting Started](#getting-started)
- [Technical Overview](#technical-overview)
  - [Database Preprocessing](#database-preprocessing)
  - [Libs](#libs)
  - [MTP-NT Compilling](#mtp-nt-compilling)
  - [Competitors Compilling](#competitors-compilling)
  - [Hourly Compilling](#hourly-compilling)
  - [Post-processing of Results](#post-processing-of-results)
- [License](#license)
  - [MIT License](#mit-license)
- [Acknowledgments](#acknowledgments)

## Introduction

The development of techniques able to forecast the mobile network traffic in a city can feed data driven applications, as VNF orchestrators, optimizing the resource allocation and increasing the capacity of mobile networks. Despite the fact that several studies have addressed this problem, many did not consider neither the traffic relationship among city regions nor information from public transport stations, which may provide useful information to better anticipate the network traffic.

In this work, we present a new deep learning architecture to forecast the network traffic using representation learning and recurrent neural networks. The framework, named MTP-NT, has two major components: the first responsible to learn from the time
series of the region to be predicted, and the second one learning from the time series of both neighboring regions and public transportation stations. The work also reviews the 5G infrastructure based on open 3GPP specifications to explore ways to implement the framework in a real architecture. Several experiments were conducted over a dataset from the city of Milan, as well as comparisons against widely adopted and state-of-the-art techniques. The results shown in this work demonstrate that the usage of public transport information contribute to improve the forecasts in central areas of the city, as well as in regions with aperiodic demands, such as tourist regions.

Thus, this research seeks to evaluate the performance of traffic forecasting models using public data, in order to validate the performance gain with the aggregation of public transport data. The aggregation of unconventional data can be a way of adding information to the model through input that has not been explored in the scope of this research area.

The development of MTP-NT was carried out during the master's program at the Federal University of Uberlândia. The slides used in the defense, presented on 11/21/2023, can be found in the file named [defesa.pdf](documentation/defesa.pdf).

## Getting Started

Before execute any of the files, please install the environment listed in ```requirements.txt``` using [pip](https://pypi.org/project/pip/) and [Anaconda](https://www.anaconda.com/).

## Technical Overview

### Database preprocessing

Before all model development, some pre work were done in the original database and in the collected data of public transport hubs.

[misc/database_adapt.py](misc/database_adapt.py): this code is used to take the original dataframe, that is in a format "one file per day" to a format "one region per day".

[transport_modelling](transport_modelling/): contains the code to map the transport hubs in Milan. The sources used was [ATM website](https://www.atm.it/en/ViaggiaConNoi/Pages/SchemaReteMetro.aspx), [Wikipedia list of Milan Metro stations](https://en.wikipedia.org/wiki/List_of_Milan_Metro_stations) and [Google Maps Platform](https://developers.google.com/maps?hl=pt-br). All data was compilled in [transport_modelling/public_transport_locations.csv](transport_modelling/public_transport_locations.csv)

- [transport_modelling/transport_locations.py](mtransport_modellingisc/transport_locations.py): takes a list of metro, tram and bus stations and, from the Google Maps API, saves the coordinates of the stations.
- [transport_modelling/transport_locations_mapping.py](transport_modelling/transport_locations_mapping.py): take the coordinates of every station and find the equivalent region on Milano Grid.

### Libs

Some code were developed to support the models training (both MTP-NT and its competitors) in different stages. They are:

Code used in model development:
 - [libs/get_milano.py](libs/get_milano.py): a library build to get the requested data from the dataset.
 - [libs/functions.py](libs/functions.py): NMAE (Normalized Mean Absolute Error) and MARE (Mean absolute Relative error) implementations.


### MTP-NT compilling

The MTP-NT is the purposed model, compilled by [model_building.py](model_building.py) script.

Some variables need to be attended to guarantee the work of the script:

- [comms_path](model_building.py#L37) needs to point to repository of the data after preprocessing by [misc/database_adapt.py](misc/database_adapt.py).
- [transport_path](model_building.py#L38) needs to point to the transport hubs data crrated by [transport_modelling/transport_locations.py](mtransport_modellingisc/transport_locations.py) and [transport_modelling/transport_locations_mapping.py](transport_modelling/transport_locations_mapping.py)

In [lines 142--178](model_building.py#L142) the region ids were the model are going to be evaluated are selected. In the end, the list of ids is stored in [ids_to_use](model_building.py#L170).

A print of the selected ids is saved in [check_selected_ids.jpg](check_selected_ids.jpg) in [line 191](model_building.py#L191).

[transport_hubs](model_building.py#L204) is a list that can control the activation of transport hubs data as well as [neighorrs](model_building.py#L205) controls wich degrees will be compilled.

After model construction and compilling, the results are saved:
- models are saved in h5 format from [lines 367--371](model_building.py#L367)
- real values and predictions are saved in csv model from [lines 381--384](model_building.py#L381)
- A plot of $y$ and $\hat{y}$ is saved in [lines 389--400](model_building.py#L389)
- The error csv is saved in [lines 403--415](model_building.py#L403)

### Competitors compilling

[model_building_ARIMA.py](model_building_ARIMA.py): constructs ARIMA models for a selected number of regions.

[model_building_HW.py](model_building_HW.py): constructs Holt-Winters models for a selected number of regions.

[model_building_LSTM.py](model_building_LSTM.py): constructs LSTM models for a selected number of regions.

[model_building_ARIMA.py](model_building_ARIMA.py): constructs ARIMA models for a selected number of regions.

[model_building_SARIMAX.py](model_building_SARIMAX.py): constructs SARIMAX models for a selected number of regions.

### Hourly compilling

The original database, after compilling as described in [Database Preprocessing](#database-preprocessing) can be recompilled again in hourly samples with the script in [misc/database_adapt_hourly.py](misc/database_adapt_hourly.py).

After all preprocessing, the resulting data also can be processed by procedures explained in [MTP-NT Compilling](#mtp-nt-compilling) and [Competidors compilling](#competidors-compilling).

### Post-processing of results

Code use to validation and compilling of results:
- [misc/compile_results.py](misc/compile_results.py): compile the results from constructed models.


## License

This project is licensed under the [Creative Commons 4.0](https://creativecommons.org/licenses/by/4.0/legalcode.en).

## Acknowledgments

Special thanks to the following contributors:

<div align="center">
  <a href="https://github.com/PatrickLdA">
    <img src="https://avatars.githubusercontent.com/u/43734047?v=4" width="100" alt="Patrick Luiz de Araújo">
  </a>
  <a href="https://www.facom.ufu.br/~pasquini/">
    <img src="https://www.facom.ufu.br/~pasquini/rafael.jpg" width="75" alt="Rafael Pasquini">
  </a>
  <a href="http://lattes.cnpq.br/8158868389973535">
    <img src="https://github.com/contributor3.png" width="100" alt="Contributor 3">
  </a>
</div>

We would also like to express our gratitude to [PPGCO-UFU](http://www.ppgco.facom.ufu.br/) for their support. And [Luis Miguel Contreras Murillo](https://es.linkedin.com/in/luis-miguel-contreras-murillo-55777b5?challengeId=AQHq9dYtn05t-QAAAYx-1BSBknavMTZWkvTf4OOP9zZ8ahcA1Jyk-Y3TgmusavZO70zIw3RBwBQl-frvBDyHqTgKKoiYnGGGLw&submissionId=e5765478-3f0a-a217-f27e-e251b45b1439&challengeSource=AgHcy8p0cBdF4wAAAYx-1B9rCBcedbowVjU_5PJEvTyAmZ4zTV_bf4ytMcOxzFg&challegeType=AgGU51MfTdoq7gAAAYx-1B9vqWF6R8SRvLkzi6R5O4ETA0drVhYo7vw&memberId=AgECj6EQ7K4GswAAAYx-1B9y_jWRfyYv8_zwvKym95TpN7A&recognizeDevice=AgEf0H9FDzVb2wAAAYx-1B91lFYZjMbM-K5hcRKLvnQCgU6eV3XZ) as supporter of the research.
