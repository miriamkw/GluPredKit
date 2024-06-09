---
title: 'GluPredKit: A Python Package for Blood Glucose Prediction and Evaluation'
tags:
  - Python
  - Blood glucose prediction
  - Non-linear Physiological modelling
  - Deep neural network
  - Machine learning
authors:
  - name: Miriam Kopperstad Wolff
    orcid: 0009-0000-6919-8164
    affiliation: 1 
  - name: Sam Royston
    orcid: 0009-0003-8623-4042
    affiliation: 2
  - name: Rune Volden
    affiliation: 1
affiliations:
 - name: Norwegian University of Science and Technology, Norway
   index: 1
 - name: Replica Health, United States of America
   index: 2
date: 24 April 2024
bibliography: paper.bib
---

# Summary

Managing blood glucose levels is crucial for individuals with diabetes. Historically, non-linear physiological modeling of glucose dynamics laid the groundwork for automated insulin delivery. Blood glucose prediction can be used as decision support for patients or as a component in an automated insulin delivery control strategy. Today, machine learning and deep neural networks offer new pathways for improvement, and the literature is vast on proposed models. Yet, comparing these advanced models is challenging. Differences in the datasets used for testing and how results are evaluated can make comparisons from existing studies unreliable [@Jacobs:2023]. Additionally, many research studies do not share their code, making it hard to build upon previous work. `GluPredKit` addresses these issues by standardizing the pipeline steps needed for any blood glucose prediction research (see \autoref{fig:flow}). This includes the collection, organization, and preparation of data, as well as the ability to easily compare different models and measure their effectiveness. Additionally, the software incorporates state-of-the-art components, including the ability to integrate and standardize data from various sources, utilize existing prediction models, and apply established evaluation metrics. It also features automated generation of detailed model evaluation reports, guided by the consensus on blood glucose model evaluation [@Jacobs:2023].


![High-level visualization of the GluPredKit ecosystem.\label{fig:flow}](GluPredKit Flow Diagram.png){ width=60% }


# Statement of need


`GluPredKit` is a Python package designed to streamline the process of blood glucose prediction, accessible via both a Command Line Interface (CLI) and as a library on PyPi. There is a need for standardized evaluation guidelines to leverage the potential of artificial intelligence in enhancing glycemic control for diabetes management [@Jacobs:2023]. Research indicates that modern deep learning models can provide superior predictions compared to traditional complex non-linear physiological models [@Cappon:2023].

`GluPredKit` addresses this need by facilitating the evaluation of individual or multiple models. Its modular design and standardized approaches facilitate community contributions, enabling researchers to integrate new models, metrics, or data sources while maintaining compatibility with existing components in a plug-and-play framework. The software includes state-of-the-art features such as integration with common data sources, ready-to-use white- and black-box models, and evaluation metrics. 

Despite the vast literature on proposed blood glucose prediction algorithms and benchmarking studies [@xie2020benchmarking], [@meijner2017blood], many do not provide open-source code or are not designed for scalability and integration with other models or data sources. The GLYFE study represents the closest existing package to GluPredKit, as it benchmarks several models and shares its source code, allowing the addition of new components [@debois2022glyfe]. However, `GluPredKit` differs from GLYFE in being more flexible in accommodating different dataset input features and hypothetical scenarios, and visualizations such as plots and predicted trajectories in addition to evaluation metrics. Researchers often need to generate these visualizations and metrics repeatedly for each experiment and research publications. Furthermore, `GluPredKit` is designed not only as a standalone package but also as a dependency that can be integrated into other software systems, unlike GLYFE's GUI-centric approach.

To ensure broad usability and scalability, `GluPredKit` consolidates prominent work in the field into a single repository with a scalable architecture that supports future community contributions. Its usability has been validated through user tests [@Wolff2024]. Additionally, Oh et al. utilized the platform in a master’s thesis, incorporating existing models from the literature and validating them against reported results [@oh2024thesis]. Integrated test datasets include the Ohio dataset [@marling2020ohiot1dm], Tidepool API, and Nightscout API. The software implements physiological models based on Uva Padova, using implementations from ReplayBG [@cappon2023replaybg] and PyLoopKit, both of which are open-source, in contrast to proprietary models in commercial systems. Moreover, off-the-shelf models such as Ridge Regressor, Random Forest, and LSTM have been implemented, based on common blood glucose prediction model approaches in benchmarks [@xie2020benchmarking], [@Cappon:2023]. The complete and evolving list of components is documented in the `GluPredKit` documentation.




# `GluPredKit` Workflow

The `GluPredKit` workflow is typically used through the CLI or as a dependency in external projects via PyPi. The first step involves parsing the input data to prepare it for processing. Users then configure the settings tailored to their specific needs before moving on to model training and testing phases. After testing, the user can generate an Excel sheet and a PDF report for a standardized evaluation report based on evaluation consensus guided by Jacobs et al. [Jacobs:2023]. 


The software consists of four key modules: data source parsers, preprocessors, prediction models, and evaluation metrics. Detailed instructions and standardized code interfaces are provided in the repository's documentation, guiding contributors on how to add and integrate their modules.






# Acknowledgements

This project is supported by the Department of ICT and Natural Sciences at the Norwegian University of Science and Technology. We acknowledge contributions from Martin Steinert, Hans Georg Schaathun, and Anders Lyngvi Fougner during the genesis of this project. 


# References








