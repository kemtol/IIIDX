package features

import (
	"fmt"
	"math"
	"sort"
	"strings"
	"time"

	"github.com/parquet-go/parquet-go"
)

type yfDaily struct {
	Date   time.Time `parquet:"date"`
	Ticker string    `parquet:"ticker"`
	Open   float64   `parquet:"open"`
	High   float64   `parquet:"high"`
	Low    float64   `parquet:"low"`
	Close  float64   `parquet:"close"`
	Volume int64     `parquet:"volume"`
}

type yfDay struct {
	date   time.Time
	open, high, low, close float64
	volume int64
}

// YfRow holds OHLCV derived features for one ticker at target date.
type YfRow struct {
	Ticker string
	Cols   map[string]float64 // column_name → value
}

// ComputeYfDaily computes OHLCV derived features from yfinance_daily.parquet.
// Applies shift(1): all rolling windows use data up to T-1.
func ComputeYfDaily(parquetPath, targetDate string) ([]YfRow, error) {
	rows, err := parquet.ReadFile[yfDaily](parquetPath)
	if err != nil {
		return nil, err
	}

	t, _ := time.Parse("2006-01-02", targetDate)
	tMinus1 := t.AddDate(0, 0, -1)
	tMinus120 := t.AddDate(0, 0, -120)

	tickerDays := make(map[string][]yfDay)
	for _, r := range rows {
		if r.Date.Before(tMinus120) || r.Date.After(tMinus1) {
			continue
		}
		t := strings.TrimSuffix(r.Ticker, ".JK")
		tickerDays[t] = append(tickerDays[t], yfDay{
			date: r.Date, open: r.Open, high: r.High,
			low: r.Low, close: r.Close, volume: r.Volume,
		})
	}

	var result []YfRow
	for ticker, days := range tickerDays {
		sort.Slice(days, func(i, j int) bool { return days[i].date.Before(days[j].date) })
		r := YfRow{Ticker: ticker, Cols: make(map[string]float64)}

		n := len(days)
		if n == 0 {
			continue
		}
		last := days[n-1]
		r.set("yf_daily_open_sum", last.open)
		r.set("yf_daily_open_mean", last.open)
		r.set("yf_daily_high_sum", last.high)
		r.set("yf_daily_high_mean", last.high)
		r.set("yf_daily_low_sum", last.low)
		r.set("yf_daily_low_mean", last.low)
		r.set("yf_daily_close_sum", last.close)
		r.set("yf_daily_close_mean", last.close)
		r.set("yf_daily_volume_sum", float64(last.volume))
		r.set("yf_daily_volume_mean", float64(last.volume))

		if n >= 2 {
			prev := days[n-2]
			r.set("yf_daily_prev_close_sum", prev.close)
			r.set("yf_daily_prev_close_mean", prev.close)
			if prev.close > 0 {
				roc := (last.close - last.open) / last.open
				r.set("yf_daily_ret_oc_sum", roc)
				r.set("yf_daily_ret_oc_mean", roc)
				rcc := (last.close - prev.close) / prev.close
				r.set("yf_daily_ret_cc_sum", rcc)
				r.set("yf_daily_ret_cc_mean", rcc)
				gap := (last.open - prev.close) / prev.close
				r.set("yf_daily_gap_open_prev_close_sum", gap)
				r.set("yf_daily_gap_open_prev_close_mean", gap)
			}
			rpct := (last.high - last.low) / last.open
			r.set("yf_daily_range_pct_sum", rpct)
			r.set("yf_daily_range_pct_mean", rpct)
			if last.close > 0 {
				to := float64(last.volume) * last.close
				r.set("yf_daily_turnover_sum", to)
				r.set("yf_daily_turnover_mean", to)
			}
		}

		// Rolling close stats
		closes := make([]float64, n)
		vols := make([]float64, n)
		ranges := make([]float64, n)
		for i := range days {
			closes[i] = days[i].close
			vols[i] = float64(days[i].volume)
			ranges[i] = (days[i].high - days[i].low) / days[i].open
		}

		// Close MA + Z-score (5/20/60)
		for _, w := range []int{5, 20, 60} {
			ma := rMean(closes, w)
			z := rZ(closes, w, ma)
			r.set(f("yf_daily_close_ma_%d_sum", w), ma)
			r.set(f("yf_daily_close_ma_%d_mean", w), ma)
			r.set(f("yf_daily_close_z_%d_sum", w), z)
			r.set(f("yf_daily_close_z_%d_mean", w), z)
		}

		// Volume MA + velocity (5/20/60)
		for _, w := range []int{5, 20, 60} {
			ma := rMean(vols, w)
			vel := velocity(vols, w)
			r.set(f("yf_daily_volume_ma_%d_sum", w), ma)
			r.set(f("yf_daily_volume_ma_%d_mean", w), ma)
			r.set(f("yf_daily_volume_velocity_%d_sum", w), vel)
			r.set(f("yf_daily_volume_velocity_%d_mean", w), vel)
		}

		// Range MA + Z-score (5/20/60)
		for _, w := range []int{5, 20, 60} {
			ma := rMean(ranges, w)
			z := rZ(ranges, w, ma)
			r.set(f("yf_daily_range_ma_%d_sum", w), ma)
			r.set(f("yf_daily_range_ma_%d_mean", w), ma)
			r.set(f("yf_daily_range_z_%d_sum", w), z)
			r.set(f("yf_daily_range_z_%d_mean", w), z)
		}

		result = append(result, r)
	}
	return result, nil
}

func (r *YfRow) set(col string, v float64) {
	if math.IsNaN(v) || math.IsInf(v, 0) {
		return
	}
	r.Cols[col] = v
}

func rMean(vals []float64, window int) float64 {
	start := max(0, len(vals)-window)
	sl := vals[start:]
	if len(sl) == 0 {
		return 0
	}
	sum := 0.0
	for _, x := range sl {
		sum += x
	}
	return sum / float64(len(sl))
}

func rZ(vals []float64, window int, ma float64) float64 {
	start := max(0, len(vals)-window)
	sl := vals[start:]
	if len(sl) < 2 || ma == 0 {
		return 0
	}
	sumSq := 0.0
	for _, x := range sl {
		diff := x - ma
		sumSq += diff * diff
	}
	std := math.Sqrt(sumSq / float64(len(sl)))
	if std == 0 {
		return 0
	}
	return (sl[len(sl)-1] - ma) / std
}

func velocity(vals []float64, window int) float64 {
	start := max(0, len(vals)-window)
	sl := vals[start:]
	if len(sl) < 2 || sl[0] == 0 {
		return 0
	}
	return (sl[len(sl)-1] - sl[0]) / sl[0]
}

func f(format string, args ...interface{}) string {
	return fmt.Sprintf(format, args...)
}
