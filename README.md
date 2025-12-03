#Import Libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from scipy.stats import poisson, lognorm, gamma

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error, r2_score

#Generate Synthetic Operational Risk Dataset

np.random.seed(42)

n = 2500

# Frequency of events per month

event_frequency = np.random.poisson(lam=3, size=n)

# Severity of losses using lognormal distribution

loss_severity = np.random.lognormal(mean=9, sigma=0.7, size=n)

# Risk drivers

system_load = np.random.uniform(20, 95, size=n)

control_weakness = np.random.binomial(1, 0.3, size=n)

vendor_issue = np.random.binomial(1, 0.25, size=n)

cyber_flag = np.random.binomial(1, 0.1, size=n)

fraud_flag = np.random.binomial(1, 0.2, size=n)

# Total loss per event

total_loss = event_frequency * loss_severity

df = pd.DataFrame({

    "event_frequency": event_frequency,
    
    "loss_severity": loss_severity,
    
    "system_load": system_load,
    
    "control_weakness": control_weakness,
    
    "vendor_issue": vendor_issue,
    
    "cyber_flag": cyber_flag,
    
    "fraud_flag": fraud_flag,
    
    "total_loss": total_loss
    
})

df.head()

#Frequency Modelling (Poisson)

lambda_est = df["event_frequency"].mean()

print("Estimated Poisson λ:", lambda_est)

#Severity Modelling (Lognormal)

shape, loc, scale = lognorm.fit(df["loss_severity"])

print("Lognormal parameters:\nShape:", shape, "\nScale:", scale)

#Operational Value-at-Risk Calculation

# Monte Carlo LDA simulation

simulations = 10000

simulated_losses = []


for _ in range(simulations):

    freq = poisson.rvs(mu=lambda_est)
    
    severities = lognorm.rvs(s=shape, scale=scale, size=freq)
    
    simulated_losses.append(sum(severities))

OpVaR_99 = np.percentile(simulated_losses, 99)

print("Operational VaR (99%):", round(OpVaR_99, 2))

#Machine Learning to Predict High-Severity Losses

features = df[["event_frequency","system_load","control_weakness",

               "vendor_issue","cyber_flag","fraud_flag"]]
               
target = df["total_loss"]

X_train, X_test, y_train, y_test = train_test_split(features, target,

                                                    test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=200, random_state=42)

model.fit(X_train, y_train)

preds = model.predict(X_test)

print("MAE:", mean_absolute_error(y_test, preds))

print("R2 Score:", r2_score(y_test, preds))

#Feature Importance

plt.figure(figsize=(8,5))

sns.barplot(x=model.feature_importances_, y=features.columns)

plt.title("Risk Driver Importance")

plt.show()
