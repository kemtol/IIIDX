package features

import (
	"math"
	"sort"
	"strings"
	"time"

	"github.com/parquet-go/parquet-go"
)

type AraHistoryRow struct {
	Date, Ticker string
	Cols         map[string]float64
}

func getAraLimitPct(refPrice float64) float64 {
	if refPrice > 5000 {
		return 0.20
	} else if refPrice > 200 {
		return 0.25
	} else if refPrice > 0 {
		return 0.35
	}
	return math.NaN()
}

func ComputeAraHistory(yfDailyPath, targetDate string) ([]AraHistoryRow, error) {
	rows, err := parquet.ReadFile[yfDailyBar](yfDailyPath)
	if err != nil {
		return nil, err
	}

	tTarget, _ := time.Parse("2006-01-02", targetDate)
	tWarmup := tTarget.AddDate(0, 0, -120)

	tickerDays := make(map[string][]yfDailyBar)
	for _, r := range rows {
		if r.Date.Before(tWarmup) || r.Date.After(tTarget) {
			continue
		}
		tk := strings.TrimSuffix(r.Ticker, ".JK")
		tickerDays[tk] = append(tickerDays[tk], r)
	}

	var result []AraHistoryRow
	araBufferPct := 0.005

	for ticker, days := range tickerDays {
		sort.Slice(days, func(i, j int) bool { return days[i].Date.Before(days[j].Date) })

		type araDay struct {
			date    time.Time
			isAra   bool
			ret     float64
		}
		var stats []araDay

		for i := 1; i < len(days); i++ {
			prev := days[i-1]
			curr := days[i]
			if prev.Close <= 0 {
				continue
			}
			ret := (curr.Close - prev.Close) / prev.Close
			limitPct := getAraLimitPct(prev.Close)
			isAra := false
			if !math.IsNaN(limitPct) && ret >= (limitPct-araBufferPct) {
				isAra = true
			}
			stats = append(stats, araDay{
				date:  curr.Date,
				isAra: isAra,
				ret:   ret,
			})
		}

		for i, currDay := range days {
			if i == 0 || currDay.Date.Format("2006-01-02") != targetDate {
				continue
			}

			// We need past stats up to i-1.
			// Since 'stats' is built from i=1 to len(days)-1, the index in stats
			// corresponding to currDay (days[i]) is i-1.
			// So we need past stats strictly before i-1.
			past := stats[:i-1]
			if len(past) == 0 {
				continue
			}

			cols := make(map[string]float64)

			// was_ara_tminus1
			last := past[len(past)-1]
			if last.isAra {
				cols["was_ara_tminus1"] = 1.0
			} else {
				cols["was_ara_tminus1"] = 0.0
			}

			// ara_count_*
			for _, w := range []int{3, 5, 10, 20, 60} {
				start := len(past) - w
				if start < 0 {
					start = 0
				}
				cnt := 0.0
				for _, p := range past[start:] {
					if p.isAra {
						cnt += 1.0
					}
				}
				// The Python code uses pandas rolling sum without min_periods=1 checking for exact window size? 
				// Wait, Python: rolling(window, min_periods=1).sum(). So it just sums whatever is available.
				if w == 3 { cols["ara_count_3d"] = cnt }
				if w == 5 { cols["ara_count_5d"] = cnt }
				if w == 10 { cols["ara_count_10d"] = cnt }
				if w == 20 { cols["ara_count_20d"] = cnt }
				if w == 60 { cols["ara_count_60d"] = cnt }
			}

			// max_return_*_tminus1
			for _, w := range []int{5, 20} {
				start := len(past) - w
				if start < 0 {
					start = 0
				}
				mx := -math.MaxFloat64
				valid := false
				for _, p := range past[start:] {
					if p.ret > mx {
						mx = p.ret
					}
					valid = true
				}
				if valid {
					if w == 5 { cols["max_return_5d_tminus1"] = mx }
					if w == 20 { cols["max_return_20d_tminus1"] = mx }
				}
			}

			// Streak features
			araStreak := 0.0
			noaraStreak := 0.0
			daysSince := math.NaN()
			lastAraRet := math.NaN()

			for _, p := range past {
				if p.isAra {
					araStreak += 1.0
					noaraStreak = 0.0
					daysSince = 0.0
					lastAraRet = p.ret
				} else {
					araStreak = 0.0
					noaraStreak += 1.0
					if !math.IsNaN(daysSince) {
						daysSince += 1.0
					}
				}
			}

			cols["consecutive_ara_streak_tminus1"] = araStreak
			cols["noara_streak_tminus1"] = noaraStreak
			if !math.IsNaN(daysSince) {
				cols["days_since_last_ara"] = daysSince
			}
			if !math.IsNaN(lastAraRet) {
				cols["last_ara_return"] = lastAraRet
			}

			result = append(result, AraHistoryRow{
				Date:   targetDate,
				Ticker: ticker,
				Cols:   cols,
			})
		}
	}
	return result, nil
}
