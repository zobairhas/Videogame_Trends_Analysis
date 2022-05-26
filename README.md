# 05022022_Final_Project

## Overview

Test

The topic for our final project involves analyzing video game trends to better understand what makes them successful upon release.

Our group quickly realized we were assembled based on our passion for video games, so picking our research topic was not so demanding. The Steam platform is the largest digital distribution platform for PC gaming, and is a fantastic resource for examining gaming trends over the last decade. We will imagine that we have been approached by a company that plans to develop and release a new title on Steam's library. Our analyses will function as a valuable asset that may inform decisions regarding the company's budget and timing of release.

## Data References

Our dataset was downloaded from Kaggle in JSON format, and provides us with various information about games offered in the Steam store.

- <https://www.kaggle.com/datasets/trolukovich/steam-games-complete-dataset>

The primary dataset provides some useful columns that we can eventually turn into features for analysis. These include any column's that reference a game's success. This can be measured by critic scores, recommendations, reviews, along with average-play time. There are categories and genres columns that we will need to decide how to use, but they stand out as potential features among the remaining columns.

Some decisions will be decided within the later stages of the project. With the data downloaded, this stage is now complete. In the next step, we'll take care of preparing and cleaning the data, readying a complete data set to use for analysis.

### Technologies Used

#### Data Cleaning and Analysis

Pandas will be used to clean the data and perform any exploratory analysis. Further analysis will be completed using Python.

#### Database Storage

PostgreSQL is the database management system that we plan to use to store our cleaned datasets.

#### Machine Learning

Scikit-learn(SKlearn) is a robust machine learning library that provides a selection of tools for statistical modeling.

We intially used a regression model to see if we could find any meaningful correlations between a set of independent and dependent variables. ```total_ratings``` was a calcuation we made by finding the average of total ratings using ```positive_ratings``` and ```negative_ratings```.


We then used an ensemble learning technique (Random forest) to solve our regression and classification problems. However, the end results were lack luster as the predicted value of the model was very low (2.3% accuracy).

We also added one new featured called ```length_of_time``` to our dataset.
```length_of_time``` was calculated by subracting todays date from the ```release_date```.

**Challenges:**
- In order to properly subtract datetime objects, the dates must first be converted to a datetime datatype. This is accomplished well in the pre-processing phase; however, due to ```release_date``` having mixed timezones, Excel automatically converts those values to just objects or strings.
- It might be worth refactoring the pre-processing functions so that it converts all date values to a single time zone first so we don't run into this issue in the future.  



#### Dashboard

Our dashboard will be made within R using the Shiny package, but is subject to change.