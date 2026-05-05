package features

import (
	"math"
	"sort"
	"strings"
	"time"

	"github.com/mmmachine/bsjp/internal/fetcher"
	"github.com/parquet-go/parquet-go"
)

type globalRow struct {
	Date   time.Time `parquet:"date"`
	Close  float64   `parquet:"close"`
	Symbol string    `parquet:"symbol"`
}

// GlobalRow holds global macro features for one date.
type GlobalRow struct {
	Date string
	// IHSG (^JKSE)
	IhsgPrevClose   float64
	IhsgPrevReturn  float64
	IhsgMA5         float64
	IhsgMA20        float64
	IhsgMA100       float64
	IhsgMA200       float64
	IhsgCloseMA5Ratio  float64
	IhsgCloseMA20Ratio float64
	// Nasdaq (^IXIC)
	NasdaqPrevReturn float64
	// Nikkei (^N225)
	NikkeiPrevReturn float64
	// VIX (^VIX)
	VixPrevClose float64
	Vix5dAvg     float64
	// USDIDR (IDR=X)
	UsdidrPrevClose  float64
	UsdidrPrevReturn float64
	Usdidr5dReturn   float64
}

// ComputeGlobal reads global_indices.parquet and computes macro features for targetDate.
// All features shifted by 1 day (no look-ahead).
// If yf1hPath is non-empty and ^JKSE daily data is missing for target date,
// a proxy IHSG close is computed from the top-50 equal-weight basket of 1h bars.
func ComputeGlobal(parquetPath, yf1hPath, targetDate string) (*GlobalRow, error) {
	rows, err := parquet.ReadFile[globalRow](parquetPath)
	if err != nil {
		return nil, err
	}

	// Organize by symbol: date → close
	type symSeries struct{ dates []time.Time; closes []float64 }
	symbols := map[string]*symSeries{
		"^JKSE":  {},
		"^IXIC":  {},
		"^N225":  {},
		"^VIX":   {},
		"IDR=X":  {},
	}

	for _, r := range rows {
		s, ok := symbols[r.Symbol]
		if !ok {
			continue
		}
		s.dates = append(s.dates, r.Date)
		s.closes = append(s.closes, r.Close)
	}

	for _, s := range symbols {
		// Sort by date
		type pair struct{ d time.Time; c float64 }
		pairs := make([]pair, len(s.dates))
		for i := range s.dates {
			pairs[i] = pair{s.dates[i], s.closes[i]}
		}
		sort.Slice(pairs, func(i, j int) bool { return pairs[i].d.Before(pairs[j].d) })
		for i := range pairs {
			s.dates[i] = pairs[i].d
			s.closes[i] = pairs[i].c
		}
	}

	t, _ := time.Parse("2006-01-02", targetDate)
	tMinus1 := t.AddDate(0, 0, -1)

	// Find index for T-1 in each series
	r := &GlobalRow{Date: targetDate}

	// IHSG
	if s := symbols["^JKSE"]; s != nil {
		idx := findIdx(s.dates, tMinus1)
		exactMatch := idx >= 0 && s.dates[idx].Truncate(24*time.Hour).Equal(tMinus1.Truncate(24*time.Hour))
		useProxy := false

		if !exactMatch && idx >= 0 && yf1hPath != "" && s.closes[idx] > 0 {
			// Walk forward from the last known ^JKSE date to find most recent day with 1h bars
			for fwd := 1; fwd <= 5; fwd++ {
				tryDate := s.dates[idx].AddDate(0, 0, fwd)
				proxy := computeIhsgProxy(yf1hPath, tryDate, s.dates[idx], s.closes[idx])
				if proxy > 0 {
					r.IhsgPrevClose = proxy
					useProxy = true
					break
				}
			}
		}

		if !useProxy && idx >= 0 {
			prevClose := s.closes[idx]
			r.IhsgPrevClose = prevClose
			if idx > 0 {
				r.IhsgPrevReturn = (prevClose - s.closes[idx-1]) / s.closes[idx-1]
			}
			r.IhsgMA5 = maAt(s.closes, idx, 5)
			r.IhsgMA20 = maAt(s.closes, idx, 20)
			r.IhsgMA100 = maAt(s.closes, idx, 100)
			r.IhsgMA200 = maAt(s.closes, idx, 200)
			ma5 := r.IhsgMA5
			ma20 := r.IhsgMA20
			if ma5 > 0 {
				r.IhsgCloseMA5Ratio = prevClose / ma5
			}
			if ma20 > 0 {
				r.IhsgCloseMA20Ratio = prevClose / ma20
			}
		}
	}

	// Nasdaq
	if s := symbols["^IXIC"]; s != nil {
		idx := findIdx(s.dates, tMinus1)
		if idx > 0 {
			r.NasdaqPrevReturn = (s.closes[idx] - s.closes[idx-1]) / s.closes[idx-1]
		}
		// Fallback: if Nasdaq daily is stale, try live 1h from Yahoo Finance
		exact := idx >= 0 && s.dates[idx].Truncate(24*time.Hour).Equal(tMinus1.Truncate(24*time.Hour))
		if !exact || idx < 0 {
			if live := fetcher.GetLastClose("^IXIC"); live > 0 && idx > 0 {
				r.NasdaqPrevReturn = (live - s.closes[idx-1]) / s.closes[idx-1]
			}
		}
	}

	// Nikkei
	if s := symbols["^N225"]; s != nil {
		idx := findIdx(s.dates, tMinus1)
		if idx > 0 {
			r.NikkeiPrevReturn = (s.closes[idx] - s.closes[idx-1]) / s.closes[idx-1]
		}
		exact := idx >= 0 && s.dates[idx].Truncate(24*time.Hour).Equal(tMinus1.Truncate(24*time.Hour))
		if !exact || idx < 0 {
			if live := fetcher.GetLastClose("^N225"); live > 0 && idx > 0 {
				r.NikkeiPrevReturn = (live - s.closes[idx-1]) / s.closes[idx-1]
			}
		}
	}

	// VIX
	if s := symbols["^VIX"]; s != nil {
		idx := findIdx(s.dates, tMinus1)
		if idx >= 0 {
			r.VixPrevClose = s.closes[idx]
			r.Vix5dAvg = maAt(s.closes, idx, 5)
		}
		exact := idx >= 0 && s.dates[idx].Truncate(24*time.Hour).Equal(tMinus1.Truncate(24*time.Hour))
		if !exact || idx < 0 {
			if live := fetcher.GetLastClose("^VIX"); live > 0 {
				r.VixPrevClose = live
			}
		}
	}

	// USDIDR
	if s := symbols["IDR=X"]; s != nil {
		idx := findIdx(s.dates, tMinus1)
		if idx >= 0 {
			r.UsdidrPrevClose = s.closes[idx]
			if idx > 0 {
				r.UsdidrPrevReturn = (s.closes[idx] - s.closes[idx-1]) / s.closes[idx-1]
			}
			start := idx - 4
			if start >= 0 {
				r.Usdidr5dReturn = (s.closes[idx] - s.closes[start]) / s.closes[start]
			}
		}
		exact := idx >= 0 && s.dates[idx].Truncate(24*time.Hour).Equal(tMinus1.Truncate(24*time.Hour))
		if !exact || idx < 0 {
			if live := fetcher.GetLastClose("IDR=X"); live > 0 && idx > 0 {
				r.UsdidrPrevClose = live
				r.UsdidrPrevReturn = (live - s.closes[idx-1]) / s.closes[idx-1]
			}
		}
	}

	return r, nil
}

func findIdx(dates []time.Time, target time.Time) int {
	targetDate := target.Truncate(24 * time.Hour)
	for i := len(dates) - 1; i >= 0; i-- {
		if dates[i].Truncate(24 * time.Hour).Equal(targetDate) {
			return i
		}
		if dates[i].Before(targetDate) {
			return i // closest ≤ target
		}
	}
	return -1
}

func maAt(closes []float64, idx, window int) float64 {
	start := idx - window + 1
	if start < 0 {
		start = 0
	}
	if start >= idx+1 {
		return 0
	}
	sum := 0.0
	for i := start; i <= idx; i++ {
		sum += closes[i]
	}
	n := float64(idx - start + 1)
	return sum / n
}

func gPtr(v float64) *float64 {
	if math.IsNaN(v) || math.IsInf(v, 0) {
		return nil
	}
	return &v
}

// findPrevIdx returns the index of the most recent date strictly before target.
func findPrevIdx(dates []time.Time, target time.Time) int {
	targetDate := target.Truncate(24 * time.Hour)
	best := -1
	for i := 0; i < len(dates); i++ {
		d := dates[i].Truncate(24 * time.Hour)
		if d.Before(targetDate) {
			best = i
		}
	}
	return best
}

// computeIhsgProxy computes an IHSG close proxy from 1h bars.
// Uses the top-50 equal-weight basket: average close of the 50 most actively traded tickers.
// Scaling: proxy = refIhsg * (targetBasketAvg / refBasketAvg)
func computeIhsgProxy(yf1hPath string, targetDate time.Time, refDate time.Time, refIhsg float64) float64 {
	type hBar struct {
		Datetime time.Time `parquet:"datetime"`
		Ticker   string    `parquet:"ticker"`
		Close    float64   `parquet:"close"`
		Volume   float64   `parquet:"volume"`
	}
	bars, err := parquet.ReadFile[hBar](yf1hPath)
	if err != nil || len(bars) == 0 {
		return 0
	}

	// Aggregate: ticker -> total turnover, last close per date
	type tickerData struct{ turnover float64; closes map[string]float64 }
	tickerMap := make(map[string]*tickerData)

	for _, b := range bars {
		d := b.Datetime.Format("2006-01-02")
		t := strings.TrimSuffix(b.Ticker, ".JK")
		td, ok := tickerMap[t]
		if !ok {
			td = &tickerData{closes: make(map[string]float64)}
			tickerMap[t] = td
		}
		td.turnover += b.Close * b.Volume
		if _, has := td.closes[d]; !has {
			td.closes[d] = b.Close
		}
	}

	// Pick top-50 by total turnover
	type kv struct{ t string; v float64 }
	var ranked []kv
	for t, td := range tickerMap {
		ranked = append(ranked, kv{t, td.turnover})
	}
	sort.Slice(ranked, func(i, j int) bool { return ranked[i].v > ranked[j].v })
	topN := 50
	if topN > len(ranked) {
		topN = len(ranked)
	}
	top50 := ranked[:topN]

	refDateStr := refDate.Format("2006-01-02")
	tgtDateStr := targetDate.Format("2006-01-02")

	var refSum, tgtSum float64
	var refCnt, tgtCnt int
	for _, tk := range top50 {
		td := tickerMap[tk.t]
		if c, ok := td.closes[refDateStr]; ok {
			refSum += c
			refCnt++
		}
		if c, ok := td.closes[tgtDateStr]; ok {
			tgtSum += c
			tgtCnt++
		}
	}

	if refCnt < 10 || tgtCnt < 10 {
		return 0
	}

	refAvg := refSum / float64(refCnt)
	tgtAvg := tgtSum / float64(tgtCnt)
	return refIhsg * (tgtAvg / refAvg)
}
