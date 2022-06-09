# <center>Predicting Tax Value of Homes using Zillow Data </center>

<img src="img/Logo-Zillow.jpeg" width=800 height=300 />

## Project Summary

### Project Objectives

- Provide a Jupyter Notebook with the following:
    - Data Acquisition
    - Data Preparation
    - Exploratory Data Analysis
        - Statistical testing where needed for exploration of data 
    - Model creation and refinement
    - Model Evaluation
    - Conclusions and Next Steps
- Creation of python modules to assist in the entire process and make it easily repeatable
    - acquire.py
    - prepare.py
    - explore.py (if necessary)
    - model.py (if necessary)
- Ask exploratory questions of the data that will give an understanding about the attributes and drivers of tax value of homes    
    - Answer questions through charts and statistical tests
- Construct the best possible model for predicting tax value of homes
    - Make predictions for a subset of out-of-sample data
- Adequately document and annotate all code
- Give a 5 minute presentation to the Zillow Data Science Team
- Field questions about my specific code, approach to the project, findings and model

### Business Goals
- Construct an ML Regression model that predict property tax assessed values ('taxvaluedollarcnt') of Single Family Properties using attributes of the properties. Namely, find a way to create a better model than the one the Data Science team already has.
- **Find the key drivers** of property value for single family properties.
- Deliver a report that the data science team can read through and replicate, understand what steps were taken, why and what the outcome was.
- Make recommendations on what works or doesn't work in prediction these homes' values.

### Audience
- Target audience for my final report is the Zillow data science team

### Project Deliverables
- A github readme explaining the project
- A jupyter notebook containing my final report (properly documented and commented)
- All modules containing functions created to achieve the final report notebook (acquire and prepare files)
- Other supplemental artifacts created while working on the project (e.g. exploratory/modeling notebook(s))
- Live presentation of final report notebook
---
### Data Dictionary

Target|Datatype|Definition|
|:-------|:--------|:----------|
| taxvaluedollarcnt | 51474 non-null: float64 | The total tax assessed value of the parcel |

The following are features I used in my final model.

|Feature|Datatype|Definition|
|:-------|:--------|:----------|
| bathroomcnt       | 51474 non-null: float16 |     Number of bathrooms in home including fractional bathrooms |
| bedroomcnt        | 51474 non-null: uint8 |     Number of bedrooms in home  |
| calculatedfinishedsquarefeet       | 51474 non-null: uint16 |    Calculated total finished living area of the home  |
| lotsizesquarefeet        | 51474 non-null: uint16 |     Area of the lot in square feet |
| yearbuilt       | 51474 non-null: float64 |     The Year the principal residence was built  | 
| county          | 51474 non-null: object  |     The county in which the residence is located; from fips data  |

---
### Questions/thoughts I have of the Data
- What features are most strongly correlated to tax value of homes?
    - Are any of these correlated to one another? Are there confounding variables? Not truly independent?
- I think lotsizesquarefeet and calculatedfinishedsquarefeet will have the strongest relationship with the target.
- I'm unsure how strongly bathroomcnt will correlate, but I suspect bedroomcnt will be relatively strong.

### Initial hypotheses

**Hypothesis 1:**<br>
alpha = .05<br>
$H_{0}$ Homes in Orange County and Ventura County have a lower or equal taxvaluedollarcnt than Los Angeles County.

$H_{a}$ Homes in Orange County and Ventura County have a higher taxvaluedollarcnt than Los Angeles County.

**Hypothesis 2:**<br>
alpha = .05<br>
$H_{0}$ There is no linear correlation between calculatedfinishedsquarefeet and taxvaluedollarcnt.

$H_{a}$ There is a linear correlation between calculatedfinishedsquarefeet and taxvaluedollarcnt.

**Hypothesis 3:**<br>
alpha = .05<br>
$H_{0}$ There is no linear correlation between yearbuilt and taxvaluedollarcnt.

$H_{a}$ There is a linear correlation between yearbuilt and taxvaluedollarcnt.

**Hypothesis 4:**<br>
alpha = .05<br>
$H_{0}$ There is no linear correlation between yearbuilt and taxvaluedollarcnt for homes in Orange and Ventura county.

$H_{a}$ There is a linear correlation between yearbuilt and taxvaluedollarcnt for homes in Orange and Ventura county.

---
## Project Plan and Data Science Pipeline

#### Plan
- **Acquire** data from the Codeup SQL Database. Create an acquire.py file containing functions to automate the process.
    - Initial inquiry into the data to see the shape and layout of things.
- Clean and **prepare** data for the explore phase. Create prepare.py to store functions I create to automate the cleaning and preparation process. Separate train, validate, test subsets and scaled data.
- Begin **exploration** of the data and ask questions leading to clarity of what is happening in the data. 
    - Clearly define at least three hypotheses, set an alpha, run any statistical tests needed, reject or fail to reject the Null Hypothesis, and document findings and takeaways.
- Create at least three different regression **models**.
    - Evaluate models on train and validate datasets. Do further hyperparamter tuning to find the best performing models.
- Choose the model with that performs the best. Do any final tweaking of the model. Automate modeling functions and put them into a model.py file.
- Evaluate final model on the test dataset.
- Construct Final Report notebook wherein I show how I arrived at the final regression model by using my created modules. Throughout the notebook, document conclusions, takeaways, and next steps.
- Create README.md with data dictionary, project and business goals, initial hypothesis and an executive summary
---
#### Plan &rarr; Acquire
- Create acquire.py to store all functions needed to acquire dataset
- Investigate the data in the Codeup SQL Database to determine what data to pull before actually acquiring the data.
- Retrieve data from the Database by running an SQL query to pull requisite Zillow data, and put it into a usable Pandas dataframe
- Do cursory data exploration/summarization to get a general feel for the data contained in the dataset
- Use the acquire.py file to import and do initial exploration/summarization of the data in the Final Report notebook
---
#### Plan &rarr; Acquire &rarr; Prepare
- Explore the data further to see where/how the data is dirty and needs to be cleaned. This is not EDA. This is exploring individual variables so as to prepare the data to undergo EDA in the next step
- Create prepare.py to store all functions needed to clean and prepare the dataset
    - A function which cleans the data:
        - Convert datatypes where necessary: objects to numerical; numerical to objects
        - Deal with missing values and nulls
        - Drop superfluous or redundant data columns
        - Handle redundant categorical variables that can be simplified
        - Change names to snake case where needed
        - Drop duplicates
    - A function which splits the dataframe into 3 subsets: Train, Validate, and Test to be used for Exploration of the data next
    - A function which creates a scaled version of the 3 subsets: Train, Validate, and Test to be used for modeling later
- Use the prepare.py file to import and do initial cleaning/preparation of the data in the Final Report notebook
---
#### Plan &rarr; Acquire &rarr; Prepare &rarr; Explore
- Do Exploratory Data Analysis of using bivariate and multivariate stats and visualizations to find interactions in the data
- Explore my key questions and discover answers to my hypotheses by running statistical analysis on data
    - Must include at least 4 visualizations and 2 statistical tests
- Find key features to use in the model. Similarly find unnecessary features which can be dropped
    - Look for correlations, relationships, and interactions between various features and the target
    - Understanding how features relate to one another will be key to understanding if certain features can or should be dropped/combined
- Document all takeaways and answers to questions/hypotheses
- Create an explore.py file which will store functions made to aid in the data exploration
- Use explore.py and stats testing in the final report notebook to show how I came to the conclusions about which data to use
---
#### Plan &rarr; Acquire &rarr; Prepare &rarr; Explore &rarr; Model
- Do any final pre-modeling data prep (drop/combine columns) as determined most beneficial from the end of the Explore phase
- Find and establish baseline RSME base on Mean and Median values of the train subset. This will give me an RSME level to beat with my models
- Create at least three regression models to predicate tax value of homes.
    - Given time attempt other models.
- For all models made, compare RSME results from train to validate
    - Look for hyperparamters that will give better results.
- Compare results from models to determine which is best. Chose the final model to go forward with
- Put all necessary functions for modeling functions into a model.py file
- Use model.py in the final report notebook to show how I reached the final model
- Having determined the best possible model, test it on out-of-sample data (the scaled test subset created in prepare) to determine accuracy
- Summarize and visualize results. Document results and takeaways
---
#### Plan &rarr; Acquire &rarr; Prepare &rarr; Explore &rarr; Model &rarr; Deliver
- After introduction, briefly summarize (give an executive summary) the project and goals before delving into the final report notebook for a walkthrough
- Do not talk through my entire process in the initial pipeline. This is only the "good stuff" to show how I arrived at the model I did
- Detail my thoughts as I was going through the process; explain the reasons for my choices
---
## Executive Summary
- The features found to be key drivers of the property value for Single Family Properties were:
    - calculatedfinishedsquarefeet and fips/county
- Notably homes in Los Angeles county were more disparate than homes on either Orange or Ventura counties. Request further time to dive deeper into the data collected for Los Angeles county to see if I can develop a better performing model.

**Discoveries and Recommendations**

- The exploration of the data and subsequent modeling show that developing individual models for smaller subsets of similar homes will result in better predictive models.
    - For the future, I would like to create models for specific subgroups: e.g.
        - 0 - 50% group 
        - 50 - 90% group
        - Top 10% group
- Consider requiring more accurate reporting on listing creation from realtors. This will allow a future analysis to determine usable features.
---
## Reproduce this project
- In order to run through this project yourself you will need your own env.py file that acquire.py will import to be able to access the Codeup database. You must have your own credentials to access the Codeup database
- All other requisite files for running the final project notebook (python files, images, csv files) are contained in this repository.
