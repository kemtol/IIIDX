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
	OvernightMA5, OvernightMA20, OvernightMA60                              float64
	OvernightPositiveRate5, OvernightPositiveRate20, OvernightPositiveRate60 float64
	GapdownFreq5, GapdownFreq20, GapdownFreq60                              float64
	GapdownSevereFreq5, GapdownSevereFreq20, GapdownSevereFreq60            float64
	GapUp2Freq5, GapUp2Freq20, GapUp2Freq60                                 float64
	GapDown2Freq5, GapDown2Freq20, GapDown2Freq60                           float64
	GapUp2Down2Edge5, GapUp2Down2Edge20, GapUp2Down2Edge60                  float64
	GapUp2FollowthroughFreq5, GapUp2FollowthroughFreq20, GapUp2FollowthroughFreq60 float64
	OvernightWorst5, OvernightWorst20, OvernightWorst60                     float64
	OvernightP10_5, OvernightP10_20, OvernightP10_60                        float64
}

type ovDay struct {
	day                            time.Time
	overnightRet, highVs, lowVs, closeVs float64
}

// ComputeOvernightAll returns overnight features matching build_overnight_history().
// Uses date-keyed sparse records with calendar-based window lookup.
func ComputeOvernightAll(yf1hPath string) ([]OvernightRow, error) {
	bars, err := parquet.ReadFile[yfBar](yf1hPath)
	if err != nil {
		return nil, err
	}

	type agg struct {
		open9, close15 float64
		high, low      float64
		ok9, ok15      bool
	}
	tmp := make(map[string]*agg)
	for _, b := range bars {
		d := b.Datetime.Format("2006-01-02")
		t := strings.TrimSuffix(b.Ticker, ".JK")
		k := d + "|" + t
		a, exist := tmp[k]
		if !exist {
			a = &agg{high: math.Inf(-1), low: math.Inf(1)}
			tmp[k] = a
		}
		h := b.Datetime.Hour()
		if h == 9 && !a.ok9 {
			a.open9 = b.Open
			a.ok9 = true
		}
		if h == 15 && !a.ok15 {
			a.close15 = b.Close
			a.ok15 = true
		}
		if h >= 9 && h <= 15 {
			if b.High > a.high {
				a.high = b.High
			}
			if b.Low < a.low {
				a.low = b.Low
			}
		}
	}

	type td struct{ date time.Time; a *agg }
	tickerDays := make(map[string][]td)
	dateFmt := "2006-01-02"
	for k, a := range tmp {
		parts := strings.SplitN(k, "|", 2)
		d, _ := time.Parse(dateFmt, parts[0])
		tickerDays[parts[1]] = append(tickerDays[parts[1]], td{d, a})
	}

	windows := []int{5, 20, 60}
	var result []OvernightRow

	for ticker, days := range tickerDays {
		sort.Slice(days, func(i, j int) bool { return days[i].date.Before(days[j].date) })

		// Build sparse shifted records using merged (inner-join) logic:
		// prev_close = last valid day's close15 (skip days without both open9 and close15)
		var ovs []ovDay
		var lastValidClose float64
		var lastValidHigh, lastValidLow float64
		hasLastValid := false

		for i := 0; i < len(days); i++ {
			curr := days[i].a
			if hasLastValid && curr.ok9 && lastValidClose != 0 {
				ret := (curr.open9 - lastValidClose) / lastValidClose
				if math.IsNaN(ret) || math.IsInf(ret, 0) {
					ret = 0
				}
				od := ovDay{day: days[i].date, overnightRet: ret}
				if curr.high != math.Inf(-1) && curr.low != math.Inf(1) {
					od.highVs = (curr.high - lastValidClose) / lastValidClose
					od.lowVs = (curr.low - lastValidClose) / lastValidClose
				}
				if curr.ok15 {
					od.closeVs = (curr.close15 - lastValidClose) / lastValidClose
				}
				ovs = append(ovs, od)
			}
			if curr.ok15 && curr.close15 != 0 {
				lastValidClose = curr.close15
				lastValidHigh = curr.high
				lastValidLow = curr.low
				hasLastValid = true
			}
		}
		_ = lastValidHigh
		_ = lastValidLow
		if len(ovs) == 0 {
			continue
		}

		// For each date, look back by POSITION (last W valid ovs entries before this date)
		for _, day := range days {
			ds := day.date.Format(dateFmt)

			// Collect ovs with date < this date, sorted
			var past []ovDay
			for _, ov := range ovs {
				if ov.day.Before(day.date) {
					past = append(past, ov)
				} else {
					break // ovs and days are both sorted
				}
			}
			if len(past) == 0 {
				continue
			}

			row := OvernightRow{Date: ds, Ticker: ticker}
			has := false

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
				ma := sum / float64(len(seg))
				posRate := float64(pos) / float64(len(seg))

				sorted := make([]float64, len(seg))
				copy(sorted, vals)
				sort.Float64s(sorted)
				p10 := sorted[max(0, int(float64(len(sorted)-1)*0.10))]
				nv := float64(len(seg))
				up2Rate := float64(up2) / nv
				down2Rate := float64(down2) / nv

				switch w {
				case 5:
					row.OvernightMA5 = f64(ma); row.OvernightPositiveRate5 = f64(posRate)
					row.GapdownFreq5 = f64(float64(gd)/nv); row.GapdownSevereFreq5 = f64(float64(gds)/nv)
					row.GapUp2Freq5 = f64(up2Rate); row.GapDown2Freq5 = f64(down2Rate)
					row.GapUp2Down2Edge5 = row.GapUp2Freq5 - row.GapDown2Freq5
					row.GapUp2FollowthroughFreq5 = f64(float64(up2Follow) / float64(max(1, up2)))
					row.OvernightWorst5 = f64(mn); row.OvernightP10_5 = f64(p10)
				case 20:
					row.OvernightMA20 = f64(ma); row.OvernightPositiveRate20 = f64(posRate)
					row.GapdownFreq20 = f64(float64(gd)/nv); row.GapdownSevereFreq20 = f64(float64(gds)/nv)
					row.GapUp2Freq20 = f64(up2Rate); row.GapDown2Freq20 = f64(down2Rate)
					row.GapUp2Down2Edge20 = row.GapUp2Freq20 - row.GapDown2Freq20
					row.GapUp2FollowthroughFreq20 = f64(float64(up2Follow) / float64(max(1, up2)))
					row.OvernightWorst20 = f64(mn); row.OvernightP10_20 = f64(p10)
				case 60:
					row.OvernightMA60 = f64(ma); row.OvernightPositiveRate60 = f64(posRate)
					row.GapdownFreq60 = f64(float64(gd)/nv); row.GapdownSevereFreq60 = f64(float64(gds)/nv)
					row.GapUp2Freq60 = f64(up2Rate); row.GapDown2Freq60 = f64(down2Rate)
					row.GapUp2Down2Edge60 = row.GapUp2Freq60 - row.GapDown2Freq60
					row.GapUp2FollowthroughFreq60 = f64(float64(up2Follow) / float64(max(1, up2)))
					row.OvernightWorst60 = f64(mn); row.OvernightP10_60 = f64(p10)
				}
				has = true
			}
			if has {
				result = append(result, row)
			}
		}
	}
	return result, nil
}

func ComputeOvernight(yf1hPath, targetDate string) ([]OvernightRow, error) {
	all, err := ComputeOvernightAll(yf1hPath)
	if err != nil {
		return nil, err
	}
	var out []OvernightRow
	for _, r := range all {
		if r.Date == targetDate {
			out = append(out, r)
		}
	}
	return out, nil
}

func f64(v float64) float64 {
	if math.IsNaN(v) || math.IsInf(v, 0) {
		return 0
	}
	return v
}
