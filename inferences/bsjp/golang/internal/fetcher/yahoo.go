package fetcher

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"time"

	"github.com/parquet-go/parquet-go"
)

const yfChartURL = "https://query2.finance.yahoo.com/v8/finance/chart/%s.JK?interval=1h&range=5d&includePrePost=false"

type YFBar struct {
	Datetime time.Time `parquet:"datetime"`
	Ticker   string    `parquet:"ticker"`
	Open     float64   `parquet:"open"`
	High     float64   `parquet:"high"`
	Low      float64   `parquet:"low"`
	Close    float64   `parquet:"close"`
	Volume   float64   `parquet:"volume"`
}

type yfResponse struct {
	Chart struct {
		Result []struct {
			Timestamp  []int64 `json:"timestamp"`
			Indicators struct {
				Quote []struct {
					Open   []float64 `json:"open"`
					High   []float64 `json:"high"`
					Low    []float64 `json:"low"`
					Close  []float64 `json:"close"`
					Volume []int64   `json:"volume"`
				} `json:"quote"`
			} `json:"indicators"`
		} `json:"result"`
	} `json:"chart"`
}

// Download1h fetches 1h bars from Yahoo Finance for the given tickers and
// appends new bars to yfinance_1h.parquet. Skips tickers that already have
// data for today.
func Download1h(tickers []string, yf1hPath string, targetDate string) (int, error) {
	today, _ := time.Parse("2006-01-02", targetDate)
	todayStart := today

	// Read existing bars to build dedup set
	existingKeys := make(map[string]bool)
	if existing, err := parquet.ReadFile[YFBar](yf1hPath); err == nil {
		for _, b := range existing {
			d := b.Datetime.Format("2006-01-02")
			key := d + "|" + b.Ticker
			existingKeys[key] = true
		}
	}

	client := &http.Client{Timeout: 30 * time.Second}
	var newBars []YFBar
	fetched := 0

	for _, t := range tickers {
		// Check if already have data for today
		checkKey := todayStart.Format("2006-01-02") + "|" + t
		if existingKeys[checkKey] {
			continue
		}

		bars, err := fetchTickerBars(client, t)
		if err != nil {
			fmt.Fprintf(os.Stderr, "  skip %s: %v\n", t, err)
			continue
		}

		for _, b := range bars {
			newBars = append(newBars, b)
			existingKeys[b.Datetime.Format("2006-01-02")+"|"+b.Ticker] = true
		}
		fetched++
		fmt.Printf("  %s: %d bars\n", t, len(bars))
		time.Sleep(200 * time.Millisecond) // rate limit
	}

	if len(newBars) == 0 {
		return 0, nil
	}

	// Append to existing parquet or create new
	var allBars []YFBar
	if existing, err := parquet.ReadFile[YFBar](yf1hPath); err == nil {
		allBars = existing
	}
	allBars = append(allBars, newBars...)

	f, err := os.Create(yf1hPath)
	if err != nil {
		return 0, fmt.Errorf("create %s: %w", yf1hPath, err)
	}
	defer f.Close()

	w := parquet.NewWriter(f, parquet.SchemaOf(YFBar{}))
	for _, b := range allBars {
		if err := w.Write(b); err != nil {
			return 0, fmt.Errorf("write: %w", err)
		}
	}
	if err := w.Close(); err != nil {
		return 0, fmt.Errorf("close: %w", err)
	}

	return len(newBars), nil
}

func fetchTickerBars(client *http.Client, ticker string) ([]YFBar, error) {
	url := fmt.Sprintf(yfChartURL, ticker)
	req, err := http.NewRequest("GET", url, nil)
	if err != nil {
		return nil, err
	}
	req.Header.Set("User-Agent", "Mozilla/5.0")

	resp, err := client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("HTTP %d", resp.StatusCode)
	}

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}

	var r yfResponse
	if err := json.Unmarshal(body, &r); err != nil {
		return nil, err
	}

	if len(r.Chart.Result) == 0 {
		return nil, fmt.Errorf("no data")
	}
	result := r.Chart.Result[0]
	if len(result.Indicators.Quote) == 0 {
		return nil, fmt.Errorf("no quotes")
	}
	q := result.Indicators.Quote[0]

	loc, _ := time.LoadLocation("Asia/Jakarta")
	var bars []YFBar
	for i, ts := range result.Timestamp {
		if i >= len(q.Open) {
			break
		}
		if isZero(q.Open[i], q.High[i], q.Low[i], q.Close[i]) {
			continue
		}
		bars = append(bars, YFBar{
			Datetime: time.Unix(ts, 0).In(loc),
			Ticker:   ticker,
			Open:     q.Open[i],
			High:     q.High[i],
			Low:      q.Low[i],
			Close:    q.Close[i],
			Volume:   float64(q.Volume[i]),
		})
	}
	return bars, nil
}

// GetLastClose fetches the latest 1h close for a Yahoo Finance symbol.
// Used as fallback when daily data is stale. Returns 0 on failure.
func GetLastClose(symbol string) float64 {
	client := &http.Client{Timeout: 10 * time.Second}
	url := fmt.Sprintf("https://query2.finance.yahoo.com/v8/finance/chart/%s?interval=1h&range=5d&includePrePost=false", symbol)
	req, _ := http.NewRequest("GET", url, nil)
	req.Header.Set("User-Agent", "Mozilla/5.0")

	resp, err := client.Do(req)
	if err != nil {
		return 0
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		return 0
	}
	body, _ := io.ReadAll(resp.Body)

	var r yfResponse
	if err := json.Unmarshal(body, &r); err != nil || len(r.Chart.Result) == 0 {
		return 0
	}
	q := r.Chart.Result[0].Indicators.Quote
	if len(q) == 0 {
		return 0
	}
	// Last non-zero close
	for i := len(q[0].Close) - 1; i >= 0; i-- {
		if q[0].Close[i] != 0 {
			return q[0].Close[i]
		}
	}
	return 0
}

func isZero(vals ...float64) bool {
	for _, v := range vals {
		if v != 0 {
			return false
		}
	}
	return true
}
