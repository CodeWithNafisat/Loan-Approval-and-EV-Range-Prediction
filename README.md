# Loan Approval and EV Range Prediction

Two supervised learning projects in one notebook, one classification and one regression, both built on small, messy tabular datasets.

The **EV range model** is the stronger result: a linear regression on five selected features predicts the driving range of electric vehicles with an R² of 0.88 and an RMSE of 32.6 km, well inside the error limit I set beforehand. The **loan approval model** is a logistic regression baseline that turned out to approve almost every application. I kept that result in because the diagnosis is the useful part: the accuracy looks fine until you compare it with what a model that always says "approved" would score.

---

## Project 1: Loan Approval (Logistic Regression)

### The problem

Predict whether a loan application is approved or rejected from basic applicant details. A lender or analyst could use this kind of model to triage applications.

### The data

The dataset has 10,000 rows and six columns: `Loan_ID`, `Gender`, `Married`, `ApplicantIncome`, `LoanAmount` and the target `Loan_Status`. Two things stood out in the audit:

- 507 incomes and 836 loan amounts were missing.
- 8,697 of the 10,000 rows were exact duplicates. After removing them, 1,303 records remained, so most of the dataset was repeated data.

### What I did

- Filled the missing incomes and loan amounts with the median. Income is heavily skewed (mean 5,311, median 3,812, maximum 81,000), so the median is the safer fill.
- Dropped the duplicates and winsorized both numeric columns at 8% per tail to limit the effect of extreme values.
- Label-encoded the categorical columns, split the data 75/25 with stratification (977 train, 326 test), and standardized the features using a scaler fitted on the training set only.
- Trained a logistic regression (`max_iter=1000`).

After cleaning, 890 applications were approved and 413 were not, so about 68% of the data is the "approved" class.

### Results

The test results look reasonable at first glance: accuracy of 0.681, precision of 0.683, recall of 0.996 and F1 of 0.810.

The confusion matrix tells a different story. The model flagged 222 approved loans correctly and missed one, but it identified none of the 103 rejected applications. It predicted "approved" for essentially every case.

<img width="1366" height="768" alt="Screenshot (46)" src="https://github.com/user-attachments/assets/3fb6f860-649f-4e1f-b24e-437a0daabfa9" />


The accuracy of 0.681 is almost exactly the share of approved loans in the test set (223 of 326, or 68.4%). So the model has not learned to separate the two outcomes, and the high recall and F1 come from approving everything, not from skill.


---

## Project 2: EV Range Prediction (Linear Regression)

### The problem

Predict the driving range (`range_km`) of an electric vehicle from its specifications. I defined success before modelling: the RMSE had to stay below 20% of the maximum range in the data.

### The data

The dataset covers 478 vehicles with 22 columns, including battery capacity, top speed, torque, efficiency, acceleration, fast-charging power, towing capacity, cargo volume, seats, drivetrain, segment and body dimensions.

### What I did

- Dropped `number_of_cells` (202 of 478 values missing) and `source_url`.
- Converted `cargo_volume_l` to numeric, which turned a few text entries into missing values.
- Filled torque, fast-charging power, towing capacity and cargo volume with medians, after checking their distributions were skewed. `fast_charge_port` was filled with its mode, CCS (476 of 477 vehicles). One row with no model name was dropped, leaving 477 vehicles.
- Winsorized the numeric columns at 8% per tail, the target included.
- Label-encoded the categorical columns, split the data 80/20 (381 train, 96 test) and standardized the features.
- Used RFE with a linear regression to pick the five most useful features: `battery_capacity_kWh`, `acceleration_0_100_s`, `fast_charge_port`, `seats` and `drivetrain`.

### Results

On the held-out test set the model reached an R² of 0.8813, an MAE of 27.4 km and an RMSE of 32.6 km. The error limit I set was 20% of the maximum range, which works out to 105 km, so the RMSE of 32.6 km clears it comfortably. Relative to the average range of about 393 km, the MAE is around 7% and the RMSE around 8% (my own calculation from the dataset's mean).

I also ran the standard checks for a linear model: scatter plots of each selected feature against range, a histogram of the residuals, and a plot of residuals against predictions.

<img width="1366" height="768" alt="Screenshot (47)" src="https://github.com/user-attachments/assets/151bf673-9f5b-450a-ab8b-2fa79dcd0de0" />

---

## What I Took From Both

- A single metric can mislead. The loan model's accuracy and F1 looked acceptable until the confusion matrix and the majority-class share showed what was happening.
- Data audits matter. Finding that 87% of the loan rows were duplicates changed the size of the real dataset from 10,000 to 1,303.
- Defining the success threshold before training, as I did for the EV model, keeps the evaluation honest.

## Tech Stack

Python, pandas, NumPy, scikit-learn (LogisticRegression, LinearRegression, RFE, StandardScaler, LabelEncoder), SciPy (winsorization), seaborn, Matplotlib and Jupyter Notebook.

## Project Structure

```
├── ScreenShots/
├── LogandLinear.ipynb
├── loan_data2.csv
├── electric_vehicles_spec_2025.csv
└── README.md
```

## How to Run

```bash
git clone hhttps://github.com/CodeWithNafisat/Loan-Approval-and-EV-Range-Prediction/tree/main

cd https://github.com/CodeWithNafisat/Loan-Approval-and-EV-Range-Prediction/tree/main

pip install pandas numpy scipy seaborn matplotlib scikit-learn jupyter

jupyter notebook LogandLinear.ipynb
```

Keep both CSV files in the same folder as the notebook, then run all the cells.
