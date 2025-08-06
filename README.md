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

##  Feature Engineering (Task 1)

* Created new interaction features:

  * `Age_Fare`: Product of Age and Fare to capture joint effect of both on survival chances.
  * `Family_Size`: Total family size onboard (`SibSp` + `Parch` + 1), as families likely influenced decisions.
  * `Fare_Per_Person`: Dividing fare by family size to normalize cost per individual.

* Generated **polynomial features** (up to degree 2) from Age and Fare using `PolynomialFeatures` to capture nonlinear relationships.

* Created **binned feature**:

  * `Age_Group`: Converted continuous Age into 4 bins — Teen, Young, Adult, Senior
  * `Age_Group_Code`: Mapped each group to a numeric code (0–3) for modeling

---

##  Model Comparison (Task 2)

Trained and compared 4 models on 60/20/20 split:

| Model         | Accuracy | Precision | Recall | F1-Score |
| ------------- | -------- | --------- | ------ | -------- |
| Logistic Reg  | 0.82     | 0.78      | 0.85   | 0.81     |
| Decision Tree | ...      | ...       | ...    | ...      |
| Random Forest | ...      | ...       | ...    | ...      |
| XGBoost       | ...      | ...       | ...    | ...      |

> Each model was evaluated using training/validation data. Metrics calculated include accuracy, precision, recall, F1-score, and confusion matrix.

---

##  Hyperparameter Tuning (Task 3)

* Used `GridSearchCV` for tuning the **Random Forest** model:

  * Parameters tuned: `n_estimators`, `max_depth`, `min_samples_split`
  * Applied 3-fold cross-validation on the training set
  * Final model was tested on the 20% holdout test set

* Result:

  * **Improved accuracy and generalization**
  * Tuning helped avoid overfitting and boosted final test performance

---

##  Model Interpretation (Task 4)

* **Feature Importance**:

  * Extracted from tree-based models (e.g., Random Forest, XGBoost)
  * Top features included `Sex`, `Fare`, `Age`, `Pclass`, and interaction features like `Age_Fare`

* **Learning Curves**:

  * Plotted training vs. validation score as dataset size increased
  * Helped confirm whether the model was underfitting or overfitting

* **Summary Report**:

  * Hyperparameter tuning increased model accuracy on the test set
  * Most surprising: Some engineered features (like `Fare_Per_Person`) provided more signal than expected
  * Final model outperformed the original baseline logistic regression model

---

##  Final Model Results

* **Best Model**: Random Forest (after tuning)
* **Final Test Accuracy**: \~XX% (replace with actual value)
* **Key Features**: Sex, Age, Pclass, Age\_Fare, Fare\_Per\_Person
* **Tools Used**: `scikit-learn`, `xgboost`, `pandas`, `matplotlib`

---

