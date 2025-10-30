"""
Event Study: Monetary Policy Announcements (RBI & Fed)
Author : Parth Kant
Date   : 2025
"""
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for saving figures
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from scipy import stats
from arch import arch_model
from datetime import timedelta
from datetime import datetime
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


# 1. Load and prepare data

print("Downloading market data...")

nifty = yf.download("^NSEI", start="2005-01-01", end="2025-01-01")
sp500 = yf.download("^GSPC", start="2005-01-01", end="2025-01-01")

for df in [nifty, sp500]:
    df["log_ret"] = np.log(df["Close"]).diff()
    df.dropna(inplace=True)

print(f"Nifty data points: {len(nifty)} | S&P500 data points: {len(sp500)}")


# 2. Read RBI & Fed event dates from CSV

events_df = pd.read_csv("data/events.csv", parse_dates=["date"])
events = events_df["date"].tolist()
print(f"Total policy events loaded: {len(events)}")

# helper to grab a time window around each event
def get_window(series, event_date, start_offset, end_offset):
    start = event_date + timedelta(days=start_offset)
    end = event_date + timedelta(days=end_offset)
    return series.loc[start:end]


# 3. Function to run event study

def run_event_study(price_series, label):
    window = range(-10, 11)
    all_ar = []

    for event in events:
        est = get_window(price_series, event, -200, -11)
        ev = get_window(price_series, event, -10, 10)
        if len(est) < 30 or len(ev) < 10:
            continue
        mu = est.mean()
        ar = ev - mu
        offsets = [(d - event).days for d in ar.index]
        ar.index = offsets
        all_ar.append(ar.reindex(window).values)

    all_ar = np.array(all_ar)
    aar = np.nanmean(all_ar, axis=0)
    acar = np.nancumsum(aar)
    print(f"{label}: used {len(all_ar)} valid events")

    return np.array(window), aar, acar, all_ar



# 4. Run for Nifty50 and S&P500

w, aar_nifty, acar_nifty, all_ar_nifty = run_event_study(nifty["log_ret"], "Nifty50")
_, aar_sp, acar_sp, all_ar_sp = run_event_study(sp500["log_ret"], "S&P500")


# 5. Plot Average and Cumulative Abnormal Returns

plt.figure(figsize=(10, 5))
plt.axvline(0, color="k", linestyle="--", label="Announcement Day")
plt.plot(w, aar_nifty, marker="o", label="Nifty50")
plt.plot(w, aar_sp, marker="s", label="S&P 500")
plt.title("Average Abnormal Return (AAR) around Monetary Policy Announcements")
plt.xlabel("Days from Event")
plt.ylabel("AAR (log return)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
# save BEFORE showing
plt.savefig(f"results/aar_plot_{timestamp}.png", dpi=300, bbox_inches='tight')
plt.close()

plt.figure(figsize=(10, 5))
plt.plot(w, acar_nifty, marker="o", label="Nifty50")
plt.plot(w, acar_sp, marker="s", label="S&P 500")
plt.title("Cumulative Average Abnormal Return (ACAR)")
plt.xlabel("Days from Event")
plt.ylabel("Cumulative Abnormal Return")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
plt.savefig(f"results/acar_plot_{timestamp}.png", dpi=300, bbox_inches='tight')
plt.close()


# 6. Simple t-test for CAR [-1,+1]

def t_test_car(all_ar):
    car_events = np.nansum(all_ar[:, 9:12], axis=1)
    t, p = stats.ttest_1samp(car_events, 0.0)
    return t, p

t_nifty, p_nifty = t_test_car(all_ar_nifty)
t_sp, p_sp = t_test_car(all_ar_sp)

print("\n=== Significance Test (CAR [-1,+1]) ===")
print(f"Nifty50 : t = {t_nifty:.2f}, p = {p_nifty:.4f}")
print(f"S&P500  : t = {t_sp:.2f}, p = {p_sp:.4f}")


# 7. Volatility analysis (GARCH example)

print("\nFitting GARCH(1,1) model on Nifty50 returns...")
am = arch_model(nifty["log_ret"] * 100, vol="Garch", p=1, q=1)
res = am.fit(disp="off")
print(res.summary())

print("\nAnalysis completed successfully!")
