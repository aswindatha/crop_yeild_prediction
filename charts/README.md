# Crop Yield Prediction Analytics Dashboard

This directory contains comprehensive analytics charts for the CropWise crop yield prediction system. These charts provide insights into model performance, feature importance, crop-specific patterns, environmental factors, and farm management impacts.

## 📊 Analytics Charts Overview

### **Category 1: Model Performance Comparison**

#### 📈 01_model_performance_comparison.png
**What it shows:** Compares the accuracy of three prediction models (MDN, Transformer, and Ensemble)

**Charts included:**
- **R² Score Comparison:** Bar chart showing how well each model predicts crop yields (higher is better)
- **Error Metrics Comparison:** Shows RMSE (Root Mean Square Error) and MAE (Mean Absolute Error) for each model (lower is better)
- **Accuracy by Crop Type:** Shows which model performs best for each crop (rice, sugarcane, cotton, etc.)
- **Regional Performance:** How well models perform in different regions (coastal, inland, hills)

**Why it matters:** Helps identify which model is most accurate and if any model has biases toward specific crops or regions.

---

### **Category 2: Prediction vs Actual Analysis**

#### 📈 02_prediction_vs_actual_analysis.png
**What it shows:** How close the predictions are to actual crop yields

**Charts included:**
- **Actual vs Predicted Scatter Plots:** Points along the diagonal line mean accurate predictions
- **Residual Plot:** Shows prediction errors (distance from the line = error magnitude)
- **Residual Distribution Histogram:** Shows if errors are normally distributed (ideal for accurate predictions)

**Why it matters:** Identifies systematic prediction errors and helps improve model accuracy.

#### 📈 03_prediction_intervals_and_errors.png
**What it shows:** Confidence intervals and error analysis

**Charts included:**
- **Prediction with 95% Confidence Intervals:** Shows the range where actual yields are likely to fall
- **Error Distribution by Yield Range:** Box plots showing errors for different yield levels (low, medium, high)
- **Q-Q Plot:** Tests if prediction errors follow a normal distribution
- **Residual Distribution with Normal Fit:** Compares actual error distribution to ideal normal distribution

**Why it matters:** Helps understand prediction uncertainty and identify if the model is overconfident or underconfident.

---

### **Category 3: Feature Importance & Impact Analysis**

#### 📈 04_feature_importance_and_impact.png
**What it shows:** Which factors most influence crop yields

**Charts included:**
- **Feature Correlation Heatmap:** Color-coded grid showing relationships between all factors
- **Feature Importance Ranking:** Bar chart of top 15 most important factors for yield prediction
- **Temperature vs Yield:** Scatter plot with trend line showing temperature impact
- **Rainfall vs Yield:** Scatter plot with trend line showing rainfall impact

**Why it matters:** Identifies which environmental factors farmers should focus on for optimal yields.

#### 📈 05_soil_properties_impact.png
**What it shows:** How soil properties affect crop yields

**Charts included:**
- **Soil pH vs Yield:** Scatter plot with optimal pH range highlighted (6.0-7.0)
- **Total Nitrogen vs Yield:** Shows nitrogen's impact on yields
- **Phosphorus vs Yield:** Shows phosphorus's impact on yields
- **CEC vs Yield:** Shows how Cation Exchange Capacity affects yields

**Why it matters:** Helps farmers understand which soil properties to test and improve for better yields.

#### 📈 06_advanced_nutrients_impact.png
**What it shows:** Impact of advanced soil nutrients on yields

**Charts included:**
- **Ammonia (NH₄⁺) vs Yield:** Shows ammonia nitrogen impact
- **Nitrate (NO₃⁻) vs Yield:** Shows nitrate nitrogen impact
- **Iron (Fe) vs Yield:** Shows iron micronutrient impact
- **Manganese (Mn) vs Yield:** Shows manganese micronutrient impact
- **Zinc (Zn) vs Yield:** Shows zinc micronutrient impact

**Why it matters:** Advanced nutrients are often overlooked but critical for optimal crop growth and yield.

---

### **Category 4: Crop-Specific Analytics**

#### 📈 07_crop_specific_yield_distribution.png
**What it shows:** Yield patterns for each crop type

**Charts included:**
- **Yield Distribution Box Plots:** Shows range, median, and outliers for each crop
- **Violin Plots:** Shows density distribution of yields for each crop
- **Average Yield by Crop:** Bar chart comparing average yields across all crops
- **Yield Statistics Table:** Summary statistics (mean, std, min, max) for each crop

**Why it matters:** Helps farmers understand expected yield ranges for different crops and set realistic expectations.

#### 📈 08_crop_regional_performance.png
**What it shows:** How crops perform in different regions

**Charts included:**
- **Crop Distribution by Region:** Stacked bar showing which crops grow in each region
- **Average Yield Heatmap:** Color-coded grid showing yield for each crop-region combination
- **Regional Yield Comparison:** Compares mean and standard deviation of yields across regions
- **Crop Count by Region:** Shows sample size of each crop in each region

**Why it matters:** Helps farmers choose crops best suited to their region and understand regional yield variations.

#### 📈 09_crop_seasonal_patterns.png
**What it shows:** Seasonal and monthly yield patterns

**Charts included:**
- **Monthly Average Yield by Crop:** Line chart showing yield trends throughout the year
- **Seasonal Average Yield:** Bar chart comparing yields across seasons (Winter, Spring, Summer, Fall)
- **Irrigation Impact by Crop:** Shows how irrigation affects different crops
- **Farm Size Impact by Crop:** Shows how farm size affects yields for different crops

**Why it matters:** Helps farmers plan planting and harvesting schedules based on seasonal patterns.

---

### **Category 5: Environmental Factor Analysis**

#### 📈 10_environmental_factors_analysis.png
**What it shows:** How weather conditions affect yields

**Charts included:**
- **Temperature vs Yield by Crop:** Scatter plots for each crop with optimal temperature range highlighted
- **Rainfall vs Yield by Crop:** Scatter plots showing rainfall impact for each crop
- **Humidity vs Yield by Crop:** Scatter plots showing humidity impact for each crop
- **Temperature Distribution by Crop:** Box plots of temperature ranges for each crop

**Why it matters:** Helps farmers understand optimal weather conditions for each crop type.

#### 📈 11_rainfall_impact_analysis.png
**What it shows:** Detailed analysis of rainfall's impact

**Charts included:**
- **Average Yield by Rainfall Range:** Bar chart showing yields for different rainfall amounts
- **Rainfall Distribution:** Histogram showing rainfall patterns in the dataset
- **Rainfall vs Yield by Region:** Regional comparison of rainfall-yield relationship
- **Average Rainfall by Region:** Bar chart comparing rainfall across regions

**Why it matters:** Helps farmers understand rainfall requirements and plan irrigation accordingly.

#### 📈 12_seasonal_pattern_analysis.png
**What it shows:** Seasonal variations in yields and weather

**Charts included:**
- **Seasonal Yield Contribution:** Pie chart showing percentage of total yield per season
- **Monthly Yield Trend:** Line chart showing yield patterns throughout the year
- **Temperature by Season:** Bar chart comparing average temperatures across seasons
- **Rainfall by Season:** Bar chart comparing average rainfall across seasons

**Why it matters:** Helps farmers optimize planting schedules based on seasonal weather patterns.

---

### **Category 6: Farm Management Analytics**

#### 📈 13_irrigation_impact_analysis.png
**What it shows:** How irrigation affects crop yields

**Charts included:**
- **Irrigation vs No Irrigation:** Bar chart comparing yields with and without irrigation
- **Yield Distribution Box Plots:** Shows yield ranges for irrigated vs non-irrigated farms
- **Irrigation Impact by Crop:** Shows which crops benefit most from irrigation
- **Irrigation Benefit Percentage:** Bar chart showing percentage yield increase from irrigation by crop

**Why it matters:** Helps farmers decide whether to invest in irrigation systems and which crops benefit most.

#### 📈 14_farm_size_analysis.png
**What it shows:** How farm size affects productivity

**Charts included:**
- **Farm Size Distribution:** Histogram showing distribution of farm sizes
- **Farm Size vs Yield:** Scatter plot with trend line showing relationship
- **Yield per Hectare by Size Category:** Bar chart showing efficiency by farm size
- **Farm Size Category Distribution:** Pie chart showing proportion of farms in each size category

**Why it matters:** Helps understand economies of scale and optimal farm sizes for maximum efficiency.

#### 📈 15_management_factors_combined.png
**What it shows:** Combined analysis of management factors

**Charts included:**
- **Yield by Farm Size and Irrigation:** Heatmap showing combined impact of size and irrigation
- **Management Efficiency:** Bar chart showing yield per hectare by size and irrigation
- **Total Yield by Farm Size Category:** Bar chart showing total production by size category
- **Farm Size Statistics Summary:** Table with comprehensive statistics for each size category

**Why it matters:** Provides holistic view of how different management factors interact to affect yields.

---

## 📋 Chart Reference Table

| Chart Number | File Name | Category | Key Insights |
|-------------|-----------|----------|--------------|
| 01 | model_performance_comparison.png | Model Performance | Model accuracy comparison |
| 02 | prediction_vs_actual_analysis.png | Prediction Analysis | Prediction accuracy and errors |
| 03 | prediction_intervals_and_errors.png | Prediction Analysis | Confidence intervals and error distribution |
| 04 | feature_importance_and_impact.png | Feature Analysis | Most important yield factors |
| 05 | soil_properties_impact.png | Feature Analysis | Soil property impacts |
| 06 | advanced_nutrients_impact.png | Feature Analysis | Advanced nutrient impacts |
| 07 | crop_specific_yield_distribution.png | Crop Analytics | Yield distribution by crop |
| 08 | crop_regional_performance.png | Crop Analytics | Regional crop performance |
| 09 | crop_seasonal_patterns.png | Crop Analytics | Seasonal yield patterns |
| 10 | environmental_factors_analysis.png | Environmental | Weather impact on yields |
| 11 | rainfall_impact_analysis.png | Environmental | Detailed rainfall analysis |
| 12 | seasonal_pattern_analysis.png | Environmental | Seasonal variations |
| 13 | irrigation_impact_analysis.png | Farm Management | Irrigation benefits |
| 14 | farm_size_analysis.png | Farm Management | Farm size impact |
| 15 | management_factors_combined.png | Farm Management | Combined management factors |

---
