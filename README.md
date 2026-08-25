# 📱 Digital Detox & Social Media Well-being Analysis

An empirical statistical study investigating the relationship between social media usage habits, FOMO (Fear of Missing Out), sleep loss, anxiety, and the effectiveness of digital detoxes. 

---

## 👥 Authors & Supervision

* **Department:** Statistics Department (English Section), Level 2 — Cairo University (FEPS)
* **Supervisor:** Dr. Sara Osama
* **Team Members:**
  * Ahmed Hossam Mohamed (Code: 5241121)
  * Ahmed Mahmoud Abdelfattah (Code: 5240136)
  * Ahmed Adel Mesbah (Code: 5240522)
  * Bassant Ahmed Abdelmeeged (Code: 5240724)

---

## 📌 Project Overview

As smartphones become integral to daily life, concerns regarding their impact on mental health, sleep quality, and anxiety have surged. While digital detoxes—taking intentional breaks from electronic devices—have grown in popularity, empirical data on their actual effectiveness remains limited. 

This project analyzes survey data to statistically test correlations between daily social media consumption, psychological stressors (FOMO, relaxation difficulty), sleep disruption, and demographic variations across gender.

---

## 📊 Key Findings

* **Sleep Loss Correlation ($p = 0.0010$):** A statistically significant moderate positive correlation ($\rho = 0.448$) exists between daily social media usage and sleep lost. The regression model indicates that **every additional hour of daily social media use costs roughly 13 minutes of lost sleep** ($y = 12.96x + 18.54$).
* **FOMO vs. Relaxation ($p = 0.7605$):** Contrary to popular assumption, there is no statistically significant correlation ($\rho = 0.044$) between high FOMO scores and the ability to relax.
* **Platform Stickiness:** **Instagram/TikTok (47.1%)** and **YouTube (33.3%)** were identified as the hardest platforms to take a break from.
* **Activity Ranks:** Users act primarily as passive consumers rather than creators. **Watching Videos** ranked highest in daily time consumption (mean rank 1.54), followed by **Messaging** (1.90), while **Posting** ranked lowest (3.29).
* **Gender Behavior:** While daily average usage time showed similar medians between genders (Males: 4h, Females: 3h), women exhibited significantly higher variance (ranging up to 8–10h). Digital detox attempts were high across both genders (70.5% Females, 61.7% Males).

---

## 🛠️ Data & Methodology

### Data Collection
Data was gathered via a structured online questionnaire covering demographics, social media addiction metrics, sleep disruption, FOMO scales, and digital detox history.

### Statistical Tests Executed
* **Spearman’s Rank Correlation ($\rho$):** Non-parametric correlation between usage time vs. sleep loss and FOMO vs. relaxation difficulty.
* **Independent Samples $t$-test / Welch’s $t$-test:** Comparing usage time and FOMO scores across genders and detox attempt groups.
* **Chi-Square Test of Independence ($\chi^2$):** Evaluating associations between categorical variables (Gender $\times$ Digital Detox Attempts).
* **One-Way ANOVA:** Testing FOMO variations across different preferred platforms.

---

## 📁 Repository Structure

```text
├── Digital Detox Survey.csv     # Cleaned survey dataset
├── Methodolgy.py                # Main Python script (Analysis & Plots)
├── Digital Detox.docx           # Full academic project report
├── requirements.txt             # Python dependencies
└── README.md                    # Project documentation
