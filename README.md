# Event Study on Monetary Policy Announcements (RBI & Fed)

This project investigates how monetary policy announcements by the **Reserve Bank of India (RBI)** and the **US Federal Reserve (Fed)** influence short-term stock market behavior.  
Using the **event study methodology**, I analyzed market reactions in **India (Nifty50)** and the **United States (S&P 500)** to identify abnormal returns and volatility patterns surrounding central bank decisions.

---

## Objective

- Measure **abnormal returns (AR)** and **cumulative abnormal returns (CAR)** around RBI and Fed policy announcements.  
- Compare the responsiveness of **emerging (Nifty50)** and **developed (S&P 500)** markets.  
- Examine post-event **volatility behavior** using a **GARCH(1,1)** model.

---

## Methodology

### 1. Data Collection
- **Market Data:**  
  - Nifty50 (`^NSEI`) and S&P 500 (`^GSPC`) historical daily prices from **Yahoo Finance** (2005–2025).  
- **Event Data:**  
  - Policy announcement dates stored in `data/events.csv` with two columns:  
    ```
    bank,date
    RBI,2023-02-08
    FED,2023-03-22
    ...
    ```

### 2. Event Study Model
- **Estimation Window:** 200 to 11 days before event (`[-200, -11]`)  
- **Event Window:** 10 days before and after event (`[-10, +10]`)  
- **Expected Return:** Mean return from estimation window.  
- **Abnormal Return (AR):**  
  \[
  AR_t = R_t - \bar{R}_{estimation}
  \]  
- Aggregated across all events to compute:
  - **AAR** (Average Abnormal Return)
  - **ACAR** (Cumulative Average Abnormal Return)

### 3. Statistical Testing
- Conducted **t-test** on CAR values over `[-1, +1]` to test significance.  
- Estimated **GARCH(1,1)** volatility model to capture post-announcement volatility clustering.

---

##  Tools and Libraries

- Python  
- pandas, numpy  
- yfinance  
- matplotlib, seaborn  
- scipy, statsmodels  
- arch (for GARCH modeling)

---

## 📊 Visual Results

**Average Abnormal Return (AAR)**
![AAR Plot](results/aar_plot.png)

**Cumulative Average Abnormal Return (ACAR)**
![ACAR Plot](results/acar_plot.png)

---

## 🔍 Key Insights

- **Nifty50** displayed **larger short-term fluctuations** around RBI announcement dates, showing more reactive price movements.  
- **S&P 500** demonstrated **smaller and quicker corrections**, consistent with higher market efficiency.  
- **GARCH results** confirmed short-term volatility spikes following both central bank announcements.  
- **Emerging markets** appear more sentiment-driven and reactive to macroeconomic communication than developed markets.

---

## 📈 Example Output

```text
=== Significance Test (CAR [-1,+1]) ===
Nifty50 : t = 2.13, p = 0.0412
S&P500  : t = 0.98, p = 0.3365

GARCH(1,1) Summary:
Volatility increased significantly during event windows for Nifty50.

