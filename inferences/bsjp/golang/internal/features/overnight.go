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
// Matches Python logic: features at date T use shift(1) of daily returns/stats.
func ComputeOvernight(yfDailyPath, targetDate string) ([]OvernightRow, error) {
	rows, err := parquet.ReadFile[yfDailyBar](yfDailyPath)
	if err != nil {
		return nil, err
	}

	t, _ := time.Parse("2006-01-02", targetDate)
	// Warmup 120 days matches Python's --warmup-calendar-days
	tWarmup := t.AddDate(0, 0, -120)

	tickerDays := make(map[string][]yfDailyBar)
	for _, r := range rows {
		// Include up to targetDate to allow shift(1) to provide T-1 data for T
		if r.Date.Before(tWarmup) || r.Date.After(t) {
			continue
		}
		tk := strings.TrimSuffix(r.Ticker, ".JK")
		tickerDays[tk] = append(tickerDays[tk], r)
	}

	windows := []int{5, 20, 60}
	var result []OvernightRow

	for ticker, days := range tickerDays {
		sort.Slice(days, func(i, j int) bool { return days[i].Date.Before(days[j].Date) })

		// Step 1: Compute daily stats (T vs T-1)
		var stats []ovDay
		for i := 1; i < len(days); i++ {
			prev := days[i-1]
			curr := days[i]
			if prev.Close == 0 {
				continue
			}
			stats = append(stats, ovDay{
				date:         curr.Date,
				overnightRet: (curr.Open - prev.Close) / prev.Close,
				highVs:       (curr.High - prev.Close) / prev.Close,
				lowVs:        (curr.Low - prev.Close) / prev.Close,
				closeVs:      (curr.Close - prev.Close) / prev.Close,
			})
		}

		// Step 2: Compute rolling features for targetDate
		// Features for targetDate T use stats from i-1 and before (shift 1)
		for i, s := range stats {
			if s.date.Format("2006-01-02") != targetDate {
				continue
			}

			// past items are stats[0...i-1]
			past := stats[:i]
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

				var sum, pos, gd, gds, up2, down2, up2Follow float64
				mn := math.MaxFloat64
				vals := make([]float64, len(seg))

				for j, ov := range seg {
					v := ov.overnightRet
					vals[j] = v
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
				posRate := pos / nv
				
				sort.Float64s(vals)
				p10 := vals[max(0, int(float64(len(vals)-1)*0.10))]

				uFreq := up2 / nv
				dFreq := down2 / nv

				suffix := ""
				switch w {
				case 5: suffix = "_5d"
				case 20: suffix = "_20d"
				case 60: suffix = "_60d"
				}

				if w == 5 || w == 20 || w == 60 {
					// Use specific names for ma and posRate to match Python's naming
					if w == 5 {
						cols["overnight_ret_ma5"] = ma
						cols["overnight_positive_rate5"] = posRate
					} else if w == 20 {
						cols["overnight_ret_ma20"] = ma
						cols["overnight_positive_rate20"] = posRate
					} else {
						cols["overnight_ret_ma60"] = ma
						cols["overnight_positive_rate60"] = posRate
					}
					
					cols["gapdown_freq"+suffix] = gd / nv
					cols["gapdown_severe_freq"+suffix] = gds / nv
					cols["gap_up2_freq"+suffix] = uFreq
					cols["gap_down2_freq"+suffix] = dFreq
					cols["gap_up2_down2_edge"+suffix] = uFreq - dFreq
					cols["gap_up2_followthrough_freq"+suffix] = up2Follow / nv
					cols["overnight_worst"+suffix] = mn
					cols["overnight_p10"+suffix] = p10
				}
			}
			if len(cols) > 0 {
				result = append(result, OvernightRow{
					Date:   targetDate,
					Ticker: ticker,
					Cols:   cols,
				})
			}
		}
	}
	return result, nil
}
