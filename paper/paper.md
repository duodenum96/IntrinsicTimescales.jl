---
title: 'IntrinsicTimescales.jl: A Julia package to estimate intrinsic (neural) timescales (INTs) from time-series data
tags:
    - Julia
    - neuroscience
    - neuroscience methods
    - intrinsic timescales
    - bayesian estimation
authors:
    - name: Yasir Çatal
      affiliation: "1, 2"
    - name: Georg Northoff
      affiliation: "1, 2"
affiliations:
    - index: 1
      name: Mind, Brain Imaging and Neuroethics Research Unit, University of Ottawa, Ontario, ON, Canada.
    - index: 2
      name: University of Ottawa Institute of Mental Health Research, Ottawa, ON, Canada.
date: 21 February 2025
bibliography: paper.bib
---

# Summary

IntrinsicTimescales.jl is a Julia package to perform estimation of intrinsic neural timescales (INTs). INTs are defined as the time window in which prior information from an ongoing stimulus can affect the processing of newly arriving information[@hasson_hierarchical_2015; @hasson_hierarchical_2008; @wolff_intrinsic_2022; @golesorkhi_brain_2021]. INTs are estimated either from the autocorrelation function (ACF) or the power spectral density (PSD) of time-series data [@honey_slow_2012; @gao_neuronal_2020]. In addition to the model-free estimates of INTs, IntrinsicTimescales.jl offers implementations of novel techniques of timescale estimation via performing parameter estimation of an Ornstein-Uhlenbeck process with adaptive approximate Bayesian computation (aABC) [@beaumont_adaptive_2009; @zeraati_flexible_2022] and automatic differentiation variational inference (ADVI) [@kucukelbir_automatic_2017]. 

# Statement of Need

Intrinsic neural timescales (INTs) were found to be an important metric to probe the brain dynamics and function. On the neuroscientific side, INTs were found to follow the large-scale gradients in the cortex ranging from uni to transmodal areas including local and long-range excitation [@murray_hierarchy_2014; @wang_macroscopic_2020] and proxies of myelination [@ito_cortical_2020; @wang_macroscopic_2020]. From a cognitive science perspective, INTs were found to be related to reward [@murray_hierarchy_2014], behavior [@zeraati_intrinsic_2023; @catal_flexibility_2024], self [@wolman_intrinsic_2023], consciousness [@zilio_intrinsic_2021] among others. Proper estimation of INTs to make sure the estimates are not affected by limited data, missingness of the data and oscillatory artifacts is crucial. While several methods exist for estimating INTs, there is a lack of standardized, open-source tools that implement both traditional model-free approaches and modern Bayesian estimation techniques. Existing software solutions are often limited to specific estimation methods, lack proper uncertainty quantification, or are not optimized for large-scale neuroimaging data.

IntrinsicTimescales.jl addresses these limitations by providing a comprehensive, high-performance toolbox for INT estimation. The package implements both established model-free methods and novel Bayesian approaches, allowing researchers to compare and validate results across different methodologies with a simple API. Its implementation in Julia ensures computational efficiency, crucial for analyzing large neuroimaging datasets. The package's modular design facilitates easy extension and integration with existing neuroimaging workflows, while its rigorous testing and documentation make it accessible to researchers across different levels of programming expertise.

# Major Features

IntrinsicTimescales.jl provides the following features:

* Model-free methods: IntrinsicTimescales.jl offers a unified API for the following INT estimation methods, making it easy for the user to compare different estimation methods and allow flexibility in scripting. 
  - ACW-50: Time to reach 0.5 in the ACF [@honey_slow_2012]
  - ACW-0: Time to reach 0.0 in the ACF [@golesorkhi_temporal_2021; @wolman_intrinsic_2023]
  - ACW-e: Time to each 1/e in the ACF [@cusinato_intrinsic_2023]
  - AUC: Area under the curve of the ACF from lag-zero to the lag where ACF reaches 0 [@manea_intrinsic_2022; @wu_mapping_2025; @watanabe_atypical_2019; @raut_organization_2020]
  - tau: The inverse decay rate of an exponential decay function fitted to the ACF [@murray_hierarchy_2014; @catal_flexibility_2024]
  - knee: Estimation of the inverse decay rate of exponential decay function by fitting a Lorentzian to the PSD. By Wiener-Khinchine theorem, the decay rate of an exponential decay function is proportional to the knee frequency of the PSD [@gao_neuronal_2020; @manea_neural_2024]

* Bayesian Parameter Estimation: These methods estimate the timescale as a parameter of a  generative model involving an Ornstein-Uhlenbeck process. Generative models support missing data and oscillatory artifacts. 
  - Adaptive Approximate Bayesian Computation (aABC) with population Monte Carlo [@beaumont_adaptive_2009; @zeraati_flexible_2022]
  - Automatic Differentiation Variational Inference (ADVI) through Turing.jl [@kucukelbir_automatic_2017]

* Specialized ACF and PSD calculations: 
  - In the case of no missing data, IntrinsicTimescales.jl calculates the ACF as the inverse fourier transform of squared magnitude of fourier transform of data for increased performance. PSD calculation is done using the periodogram method with a hamming window. In the case of missing data, ACF is calculated in time domain while accounting for the missingness of the data and PSD is calculated with the Lomb-Scargle periodogram. 

* Visualization: 
  - IntrinsicTimescales.jl offers built-in plotting utilities for ACFs and PSDs. For Bayesian methods, plots of posterior predictive checks are available. 

# Documentation


IntrinsicTimescales.jl provides comprehensive documentation that includes detailed API references and practical tutorials. The documentation is structured in three main sections: 1) Practice tutorials that build up understanding from basic concepts to advanced methods, 2) Implementation details for both model-free and Bayesian methods, and 3) Complete API reference. Each estimation method is thoroughly documented with mathematical formulations, example code and explanations of parameters. The documentation includes interactive examples demonstrating proper usage of the package's features, from basic timescale estimation to advanced Bayesian inference techniques. All documentation is hosted online and integrated with Julia's built-in help system.

IntrinsicTimescales.jl aims to become a standard tool in neuroscience research by providing robust, efficient, and well-documented methods for estimating intrinsic neural timescales across different experimental paradigms and data conditions.
