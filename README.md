# Steel Industry Load Classification

A Streamlit dashboard that classifies steel-industry energy load as **Light**, **Medium** or **Maximum** using a Random Forest classifier, and reports the model's confidence in its own answer.

## How it works

Enter nine interval measurements, press **Predict Load Type**, and the app returns the predicted class with a confidence percentage, a plain-language confidence band, and the probability assigned to each of the three classes.

Confidence is the forest's vote share. Each of the 100 trees votes on a class, and the reported percentage is how many trees agreed, which is what `predict_proba` returns. It always sums to 100% across the three classes.

## Model

Random Forest Classifier, 100 estimators, `max_features='sqrt'`, `max_depth=18`.

- Holdout accuracy (80/20 stratified split): **91.07%**
- Per-class F1: Light 0.977, Medium 0.843, Maximum 0.836

Medium and Maximum overlap in the source data, which is why their scores trail Light.

## Model loading

At startup the app tries `model.pkl` first. If that file is absent or unreadable, it trains an identical model from `Steel_industry_data.csv`, which takes about a second. The active model file is named in the sidebar.

The repository does not ship a model file. Training on startup keeps the repository small and avoids storing a large binary in git. To use a model trained elsewhere, commit it as `model.pkl` and it will be picked up automatically.

## Class mapping

| Code | Load Type |
| ---- | --------- |
| 1 | Light Load |
| 2 | Medium Load |
| 3 | Maximum Load |

## Features

| Feature | Description |
| ------- | ----------- |
| `Usage_kWh` | Total energy consumed in the interval |
| `Lagging_Current_Reactive.Power_kVarh` | Lagging reactive power drawn |
| `Leading_Current_Reactive_Power_kVarh` | Leading reactive power drawn |
| `CO2(tCO2)` | Carbon dioxide produced |
| `Lagging_Current_Power_Factor` | Lagging power factor |
| `Leading_Current_Power_Factor` | Leading power factor |
| `NSM` | Seconds elapsed since midnight |
| `Day_factorized` | Day code, 1 = Monday through 7 = Sunday |
| `WeekStatus_factorized` | Day type, 1 = Weekday, 2 = Weekend |

## Running locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Reading the output

A high confidence percentage is not a guarantee. The more useful signal is the gap between the top two probabilities. A split of 0.93 / 0.02 / 0.05 is a decisive call, while 0.55 / 0.45 means the model is nearly undecided despite naming a winner.

## Source data

`Steel_industry_data.csv`, 35,040 fifteen-minute intervals from 2018, from the UCI Steel Industry Energy Consumption dataset.