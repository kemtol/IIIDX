package features

import (
	"fmt"
	"math"
	"sort"
	"strings"
	"time"

	"github.com/parquet-go/parquet-go"
)

type yfBar struct {
	Datetime time.Time `parquet:"datetime"`
	Ticker   string    `parquet:"ticker"`
	Open     float64   `parquet:"open"`
	High     float64   `parquet:"high"`
	Low      float64   `parquet:"low"`
	Close    float64   `parquet:"close"`
	Volume   int64     `parquet:"volume"`
}

type MomentumRow struct {
	Date           string
	Ticker         string
	EntryPrice     *float64
	CloseRetLast1h *float64
	CloseVsOpenDay *float64
	CloseRangePct  *float64
}

// ComputeMomentum reads yfinance_1h.parquet and computes closing session momentum features.
// Exact match to build_closing_momentum() in generate_datamart.py.
//
//	close_ret_last1h  = (close_15 - close_14) / close_14
//	close_vs_open_day = (close_15 - open_9)  / open_9
//	close_range_pct   = (max(high, 9-15) - min(low, 9-15)) / open_9
//
// targetDate is YYYY-MM-DD. Set to "" to compute for ALL dates.
func ComputeMomentum(parquetPath, targetDate string) ([]MomentumRow, error) {
	bars, err := parquet.ReadFile[yfBar](parquetPath)
	if err != nil {
		return nil, fmt.Errorf("read momentum parquet: %w", err)
	}

	type trio struct {
		open9   float64
		close14 float64
		close15 float64
		dayHigh float64
		dayLow  float64
		ok9     bool
		ok14    bool
		ok15    bool
	}
	// Key: date|ticker
	acc := make(map[string]*trio)

	for _, b := range bars {
		dateVal := b.Datetime.Format("2006-01-02")
		if targetDate != "" && dateVal != targetDate {
			continue
		}
		ticker := strings.TrimSuffix(b.Ticker, ".JK")
		key := dateVal + "|" + ticker
		t, ok := acc[key]
		if !ok {
			t = &trio{dayHigh: math.Inf(-1), dayLow: math.Inf(1)}
			acc[key] = t
		}
		hour := b.Datetime.Hour()
		if hour == 9 && !t.ok9 {
			t.open9 = b.Open
			t.ok9 = true
		}
		if hour == 14 && !t.ok14 {
			t.close14 = b.Close
			t.ok14 = true
		}
		if hour == 15 && !t.ok15 {
			t.close15 = b.Close
			t.ok15 = true
		}
		if hour >= 9 && hour <= 15 {
			if b.High > t.dayHigh {
				t.dayHigh = b.High
			}
			if b.Low < t.dayLow {
				t.dayLow = b.Low
			}
		}
	}

	var results []MomentumRow
	for key, t := range acc {
		parts := strings.SplitN(key, "|", 2)
		dateVal, ticker := parts[0], parts[1]

		row := MomentumRow{Date: dateVal, Ticker: ticker}
		if t.ok15 && t.close15 != 0 {
			row.EntryPrice = f64p(t.close15)
		}

		if t.ok14 && t.ok15 && t.close14 != 0 {
			row.CloseRetLast1h = f64p((t.close15 - t.close14) / t.close14)
		}
		if t.ok9 && t.ok15 && t.open9 != 0 {
			row.CloseVsOpenDay = f64p((t.close15 - t.open9) / t.open9)
		}
		if t.ok9 && t.open9 != 0 && t.dayHigh != math.Inf(-1) && t.dayLow != math.Inf(1) {
			row.CloseRangePct = f64p((t.dayHigh - t.dayLow) / t.open9)
		}
		results = append(results, row)
	}
	sort.Slice(results, func(i, j int) bool {
		if results[i].Date != results[j].Date {
			return results[i].Date < results[j].Date
		}
		return results[i].Ticker < results[j].Ticker
	})
	return results, nil
}

func f64p(v float64) *float64 {
	if math.IsNaN(v) || math.IsInf(v, 0) {
		return nil
	}
	return &v
}
