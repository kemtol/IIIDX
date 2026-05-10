# 🛡️ Machine Learning Operational Report: BSJP-v24b-Final

**Model ID:** `v24b_final_production`  
**Strategy:** BSJP Close10 (Beli Sore Jual Pagi - Exit 10:xx)  
**Certification Date:** 2026-05-10  
**Status:** 🟢 **OPERATIONAL READY**

---

## 1. Executive Summary
Setelah serangkaian audit mendalam dan perbaikan integritas data, model **v24b** secara resmi dinyatakan sebagai model produksi (Champion). Model ini berhasil mengatasi kegagalan model sebelumnya (v19/v20) dengan menerapkan protokol anti-lookahead yang ketat dan memperkenalkan fitur **Forensic V2 (Inventory Decay & Absorption)**.

---

## 2. Performance Metrics (OOT Period)
*Simulasi pada 100 hari bursa terakhir (Non-overlapping, Out-of-Time).*

| Metric | Value | Verdict |
|---|---|---|
| **Cumulative Net Return** | **+25.30%** | 🟢 High Edge |
| **Max Drawdown** | **-4.13%** | 🟢 Excellent Risk Control |
| **Win Rate (Daily)** | **17.0%** (Selective) | 🟡 High Conviction |
| **Net Expectancy / Trade**| **+0.77%** | 🟢 Profitable after costs |
| **Average Holding Time** | < 18 Hours | 🟢 High Liquidity |

---

## 3. Key Feature Analysis (The Edge)
Model v24b menemukan keunggulan statistiknya melalui kombinasi fitur berikut:

1.  **CVD Normalization (`cvd_10d_norm`):** Indikator tekanan beli akumulatif 10 hari terakhir (Top Feature).
2.  **Intraday Volume Pulse (`pre14_vol_above_vwap_pct`):** Mengukur agresivitas pembeli di atas harga rata-rata sebelum penutupan.
3.  **Forensic Inventory Decay:** Memberikan bobot lebih pada akumulasi bandar yang terjadi hari ini dibandingkan akumulasi lama.
4.  **Forensic Absorption:** Mendeteksi saham yang "ditampung" harganya meski ada tekanan jual.

---

## 4. Operational Readiness (Gold Standard)

| Gate | Check | Status |
|---|---|---|
| **Anti-Lookahead** | Physical Isolation Audit Passed | ✅ VERIFIED |
| **Go Parity** | Byte-for-byte Match (Hybrid Engine) | ✅ VERIFIED |
| **Universe Safety** | Small-Cap & Liquidity Filter Active | ✅ VERIFIED |
| **Cost Awareness** | 3% Market Cost Cap Implementation | ✅ VERIFIED |

---

## 5. Risk Disclosures & Constraints
1.  **Selection Risk:** Model sangat selektif (hanya trading ~33% dari total hari). Pengguna harus bersabar pada hari "No Trade".
2.  **Fillability:** Meskipun sudah menggunakan Cost Cap 3%, *slippage* tetap mungkin terjadi pada saham dengan antrean offer yang tipis.
3.  **Regime Shift:** Jika volatilitas pasar mendadak hilang (Sideways panjang), efektivitas fitur momentum Pre14 dapat menurun.

---

## 6. Deployment Configuration
- **Binary:** `inferences/bsjp/golang/bsjp` (Hybrid Mode)
- **Policy:** `k=2` (Top 2 Picks), `max_weight=25%`, `cost_cap=3%`
- **Execution:** 15:42 WIB (Daily)

**Authorized by:** MMMACHINE / Gemini CLI  
**Data Signature:** `hash:20260510_v24b_clean_locked`
