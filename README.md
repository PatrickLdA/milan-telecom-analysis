# MTP-NT: A Mobile Traffic Predictor Enhanced by Neighboring and Transportation Data

<p align="center">
<img src="images/bing_framework_representation.jpg" alt="MTP-NT by Bing" width="400"/>
</p>

These code are a technical analisys of [A multi-source dataset of urban life in the city of Milan and the Province of Trentino](https://www.nature.com/articles/sdata201555) paper and the development of a predictive model to forecast network traffic. The work was carried out during the master's program at the Federal University of Uberlândia.


## Table of Contents

- [Introduction](#introduction)
- [Getting Started](#getting-started)
- [Technical Overview](#technical-overview)
- [License](#license)
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

### Preprocessing of database

```database_adapt.py```: this code is used to take the original dataframe, that is in a format "one file per day" to a format "one region per day".

*transport_modelling*: contains the code to map the transport hubs in Milan. The sources used was [ATM website](https://www.atm.it/en/ViaggiaConNoi/Pages/SchemaReteMetro.aspx), [Wikipedia list of Milan Metro stations](https://en.wikipedia.org/wiki/List_of_Milan_Metro_stations) and [Google Maps Platform](https://developers.google.com/maps?hl=pt-br).
- transport_locations.py: takes a list of metro, tram and bus stations and, from the Google Maps API, saves the coordinates of the stations.
- transport_locations_mapping.py: take the coordinates of every station and find the equivalent region on Milano Grid.

### Misc

*libs/*
 - get_milano.py: a library build to get the requested data from the dataset.
 - functions.py: NMAE (Normalized Mean Absolute Error) and MARE (Mean absolute Relative error) implementations.

```compile_results.py```: compile the results from constructed models.


### MTP-NT compilling

 ```model_building.py```: constructs the purposed model framework for a selected number of regions.

### Competitors compilling

```model_building_ARIMA.py```: constructs ARIMA models for a selected number of regions.

```model_building_HW.py```: constructs Holt-Winters models for a selected number of regions.

### Hourly compilling


## License

This project is licensed under the MIT License.

### MIT License

MIT License

Copyright (c) [2023] [Patrick Luiz de Araújo]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

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
