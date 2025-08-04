# Titanic-Survival-Prediction
##  Dataset Overview

* **Total entries**: 891
* **Target column**: `Survived` (0 = No, 1 = Yes)
* **Features**: Age, Fare, Pclass, SibSp, Parch, Sex, Embarked, etc.

---

##  Data Cleaning

* Handled missing values in `Age` and `Embarked`
* Converted categorical columns:

  * `Sex`: Male/Female → 0/1
  * `Embarked`: One-hot encoded into `Embarked_C`, `Embarked_Q`, `Embarked_S`
* Normalized continuous values (`Age`, `Fare`)

---

##  Exploratory Data Analysis (EDA)

###  Survival Rate

* **Overall survival rate**: \~38%
* Females had a much higher survival rate than males

###  Passenger Class

* 1st class passengers had the highest survival rate
* 3rd class had the lowest

###  Age Distribution

* Most passengers were between **20–40 years old**
* Many children under 10 survived (likely due to priority)

###  Fare

* Survivors tended to pay higher fares
* Fare distribution is right-skewed

###  Embarkation Port

* Most passengers boarded from `S`
* Highest survival rate seen among passengers from `C`

---

##  Model Summary

* Model used: **Logistic Regression**
* Accuracy: \~78%
* ROC AUC Score: \~0.84
* Model learns survival based on age, sex, class, fare, and more

---

##  Key Insights

* **Sex and class** were the strongest indicators of survival
* Younger age and higher fare also increased chances
* Machine learning models like Logistic Regression can effectively classify survival using just a few features

---
