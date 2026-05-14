# GeoAI Urban Waterlogging Dynamic Risk Assessment Model

## Overview
This repository contains the official open-source implementation of the master's thesis **"Dynamic Assessment of Urban Waterlogging Risk Driven by Multi-Source Geospatial Data and GeoAI"** from Beijing Normal University, School of Geography. It provides a complete technical framework for high spatiotemporal resolution urban waterlogging risk dynamic assessment, addressing the core limitations of traditional static assessment methods in capturing the spatiotemporal heterogeneity of urban hydrological processes and human activities.

Based on the international standard Hazard-Exposure-Vulnerability (H-E-V) risk assessment framework, this work integrates multi-source geospatial big data and geospatial artificial intelligence (GeoAI) technologies to achieve **hourly temporal resolution and 250m grid spatial resolution** full-process dynamic risk assessment. The model system has been validated using historical flood events in the core functional area of Beijing (Dongcheng and Xicheng Districts) and demonstrates high accuracy and practical applicability.

## Core Technical Architecture
The model system consists of three tightly coupled technical modules, each designed to solve specific scientific challenges in urban waterlogging risk assessment:

### 1. Multi-Contextual urban Waterlogging Dynamic Hazard Assessment Model (MTW-DHAM)
- **Technical Highlights**: Integrates four sub-modules: OSM waterway zoning clipping, locally improved Chicago rainfall pattern generation, multi-factor corrected SCS-CN runoff generation, and dynamic terrain-based physical flow accumulation
- **Key Innovations**: 
  - Adopts dynamic terrain improved D8 flow direction algorithm to solve the problem of pseudo-flow direction in flat urban built-up areas
  - Implements rainfall scenario-adaptive CN value correction mechanism
  - Supports 6-level rainfall scenario simulation from light rain to extreme rainstorm
  - Constructs hazard index combining water depth and flow velocity for comprehensive disaster intensity quantification

### 2. People-Facility Dynamic Exposure Assessment Model (PF-DEAM)
- **Technical Highlights**: Dual-dimensional exposure assessment framework integrating human life exposure and economic asset exposure
- **Key Innovations**:
  - Proposes DST-Transformer (Decoupled Spatiotemporal Transformer) model for high-precision hourly population distribution prediction
  - Introduces rainfall intensity-dependent building shelter effect correction to avoid overestimation of population exposure
  - Builds time-varying facility exposure weight system based on industrial operation rhythms (workday/weekend, day/night)

### 3. People-Facility Dynamic Vulnerability Assessment Model (PF-DVAM)
- **Technical Highlights**: Clear boundary definition between exposure and vulnerability indicators
- **Key Innovations**:
  - Implements age-gender weighted population vulnerability calculation based on disaster self-rescue ability differences
  - Constructs function-flood resistance collaborative facility vulnerability assessment method
  - Supports spatiotemporal dynamic characterization of vulnerability across four typical time scenarios

## Tech Stack
- **Deep Learning**: PyTorch (DST-Transformer spatiotemporal prediction model)
- **Spatial Analysis**: GeoPandas, GDAL, Rasterio, ArcPy
- **Statistical Analysis**: NumPy, Pandas, GeoDetector (spatial heterogeneity analysis)
- **Visualization**: Matplotlib, Seaborn
- **Data Processing**: Scikit-learn, SciPy

## Key Functionalities
✅ Multi-scenario urban waterlogging hazard dynamic simulation (6 rainfall levels, hourly step)
✅ Population exposure prediction with building shelter effect correction
✅ Time-varying facility asset exposure quantification based on sectoral GDP
✅ Age-gender structured population vulnerability assessment
✅ Function-flood resistance integrated facility vulnerability evaluation
✅ Spatiotemporally coordinated comprehensive risk calculation under H-E-V framework
✅ Geographical detector-based risk influencing factor analysis (single factor + interaction effect)
✅ Standardized multi-source geospatial data preprocessing pipeline

## Study Area & Validation
- **Study Area**: Dongcheng and Xicheng Districts, Beijing (core functional area of the capital, 92.5 km²)
- **Validation**: Verified using 4 typical historical rainstorm events in Beijing from 2021 to 2023, including the "23·7" extreme rainstorm disaster
- **Performance**: Achieved over 85% spatial matching accuracy between simulated high-risk areas and actual disaster-affected areas

## Usage & Application
This codebase can be directly used to reproduce all core experimental results of the thesis. It can also be migrated and extended to other urban central areas for fine-grained urban waterlogging risk assessment, supporting:
- Urban flood control and disaster reduction planning
- Sponge city construction effect evaluation
- Extreme rainfall emergency response decision support
- Urban resilience assessment and optimization

## Citation
If you use this code or dataset in your research, please cite the original thesis:
```
Chen, L. (2026). Dynamic Assessment of Urban Waterlogging Risk Driven by Multi-Source Geospatial Data and GeoAI. Master's Thesis, Beijing Normal University.
```
