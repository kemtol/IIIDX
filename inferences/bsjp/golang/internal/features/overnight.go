package features

import (
	"math"
	"sort"
	"strings"
	"time"

	"github.com/parquet-go/parquet-go"
)

type OvernightRow struct {
	Date, Ticker string
	Cols         map[string]float64
}

type ovDay struct {
	date                              time.Time
	overnightRet, highVs, lowVs, closeVs float64
}

type yfDailyBar struct {
	Date   time.Time `parquet:"date"`
	Ticker string    `parquet:"ticker"`
	Open   float64   `parquet:"open"`
	High   float64   `parquet:"high"`
	Low    float64   `parquet:"low"`
	Close  float64   `parquet:"close"`
}

// ComputeOvernight computes overnight history features from yfinance_daily.parquet.
// Uses shift(1): all rolling windows use data up to T-1.
func ComputeOvernight(yfDailyPath, targetDate string) ([]OvernightRow, error) {
	rows, err := parquet.ReadFile[yfDailyBar](yfDailyPath)
	if err != nil {
		return nil, err
	}

	t, _ := time.Parse("2006-01-02", targetDate)
	tMinus120 := t.AddDate(0, 0, -120)

	tickerDays := make(map[string][]yfDailyBar)
	for _, r := range rows {
		if r.Date.Before(tMinus120) || r.Date.After(t) {
			continue
		}
		tk := strings.TrimSuffix(r.Ticker, ".JK")
		tickerDays[tk] = append(tickerDays[tk], r)
	}

	windows := []int{5, 20, 60}
	var result []OvernightRow

	for ticker, days := range tickerDays {
		sort.Slice(days, func(i, j int) bool { return days[i].Date.Before(days[j].Date) })

		var ovs []ovDay
		for i := 1; i < len(days); i++ {
			prev := days[i-1]
			if prev.Close == 0 {
				continue
			}
			ret := (days[i].Open - prev.Close) / prev.Close
			if math.IsNaN(ret) || math.IsInf(ret, 0) {
				ret = 0
			}
			ovs = append(ovs, ovDay{
				date:         days[i].Date,
				overnightRet: ret,
				highVs:       (days[i].High - prev.Close) / prev.Close,
				lowVs:        (days[i].Low - prev.Close) / prev.Close,
				closeVs:      (days[i].Close - prev.Close) / prev.Close,
			})
		}
		if len(ovs) == 0 {
			continue
		}

		for _, day := range ovs {
			ds := day.date.Format("2006-01-02")
			if ds != targetDate {
				continue
			}

			var past []ovDay
			for _, ov := range ovs {
				if ov.date.Before(day.date) {
					past = append(past, ov)
				}
			}
			if len(past) == 0 {
				continue
			}

			cols := make(map[string]float64)
			for _, w := range windows {
				mp := max(3, w/4)
				start := len(past) - w
				if start < 0 {
					start = 0
				}
				seg := past[start:]
				if len(seg) < mp {
					continue
				}

				sum, pos, gd, gds := 0.0, 0, 0, 0
				mn := math.MaxFloat64
				up2, down2, up2Follow := 0, 0, 0
				vals := make([]float64, len(seg))
				for i, ov := range seg {
					v := ov.overnightRet
					vals[i] = v
					sum += v
					if v > 0 {
						pos++
					}
					if v < -0.02 {
						gd++
					}
					if v < -0.05 {
						gds++
					}
					if v < mn {
						mn = v
					}
					if ov.highVs >= 0.02 {
						up2++
						if ov.closeVs > 0 {
							up2Follow++
						}
					}
					if ov.lowVs <= -0.02 {
						down2++
					}
				}
				nv := float64(len(seg))
				ma := sum / nv
				posRate := float64(pos) / nv

				sorted := make([]float64, len(seg))
				copy(sorted, vals)
				sort.Float64s(sorted)
				p10 := sorted[max(0, int(float64(len(sorted)-1)*0.10))]

				up2Rate := float64(up2) / nv
				down2Rate := float64(down2) / nv

				switch w {
				case 5:
					cols["overnight_ret_ma5"] = f64(ma)
					cols["overnight_positive_rate5"] = f64(posRate)
					cols["gapdown_freq_5d"] = f64(float64(gd) / nv)
					cols["gapdown_severe_freq_5d"] = f64(float64(gds) / nv)
					cols["gap_up2_freq_5d"] = f64(up2Rate)
					cols["gap_down2_freq_5d"] = f64(down2Rate)
					cols["gap_up2_down2_edge_5d"] = f64(up2Rate - down2Rate)
					cols["gap_up2_followthrough_freq_5d"] = f64(float64(up2Follow) / nv)
					cols["overnight_worst_5d"] = f64(mn)
					cols["overnight_p10_5d"] = f64(p10)
				case 20:
					cols["overnight_ret_ma20"] = f64(ma)
					cols["overnight_positive_rate20"] = f64(posRate)
					cols["gapdown_freq_20d"] = f64(float64(gd) / nv)
					cols["gapdown_severe_freq_20d"] = f64(float64(gds) / nv)
					cols["gap_up2_freq_20d"] = f64(up2Rate)
					cols["gap_down2_freq_20d"] = f64(down2Rate)
					cols["gap_up2_down2_edge_20d"] = f64(up2Rate - down2Rate)
					cols["gap_up2_followthrough_freq_20d"] = f64(float64(up2Follow) / nv)
					cols["overnight_worst_20d"] = f64(mn)
					cols["overnight_p10_20d"] = f64(p10)
				case 60:
					cols["overnight_ret_ma60"] = f64(ma)
					cols["overnight_positive_rate60"] = f64(posRate)
					cols["gapdown_freq_60d"] = f64(float64(gd) / nv)
					cols["gapdown_severe_freq_60d"] = f64(float64(gds) / nv)
					cols["gap_up2_freq_60d"] = f64(up2Rate)
					cols["gap_down2_freq_60d"] = f64(down2Rate)
					cols["gap_up2_down2_edge_60d"] = f64(up2Rate - down2Rate)
					cols["gap_up2_followthrough_freq_60d"] = f64(float64(up2Follow) / nv)
					cols["overnight_worst_60d"] = f64(mn)
					cols["overnight_p10_60d"] = f64(p10)
				}
			}
			if len(cols) > 0 {
				result = append(result, OvernightRow{Date: ds, Ticker: ticker, Cols: cols})
			}
		}
	}
	return result, nil
}

func f64(v float64) float64 {
	if math.IsNaN(v) || math.IsInf(v, 0) {
		return 0
	}
	return v
}
