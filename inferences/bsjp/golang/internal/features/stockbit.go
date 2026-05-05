package features

import (
	"database/sql"
	"fmt"
	"os"
)

func toFloat(v interface{}) (float64, bool) {
	switch x := v.(type) {
	case float64:
		return x, true
	case float32:
		return float64(x), true
	case int64:
		return float64(x), true
	case nil:
		return 0, false
	}
	return 0, false
}

// XlRow holds Stockbit (broker "XL") features for one ticker at target date.
type XlRow struct {
	Ticker string
	Cols   map[string]float64
}

// ComputeStockbit computes Stockbit/XL broker features from broksum_bybroker.parquet.
// Features: xl_buy_freq_ma5, xl_sell_freq_ma5, xl_buy_freq_ma20, xl_freq_surge.
// All rolling windows use shift(1): data up to T-1 only (no lookahead).
func ComputeStockbit(db *sql.DB, broksumPath, targetDate string) ([]XlRow, error) {
	if _, err := os.Stat(broksumPath); os.IsNotExist(err) {
		return nil, fmt.Errorf("broksum_bybroker.parquet not found: %s", broksumPath)
	}

	// Step 1: load XL broker data
	sql := fmt.Sprintf(`
		CREATE OR REPLACE TABLE _xl AS
		SELECT date, stock_code AS ticker, buy_freq, sell_freq
		FROM read_parquet('%s')
		WHERE broker = 'XL'
	`, broksumPath)
	if _, err := db.Exec(sql); err != nil {
		return nil, fmt.Errorf("_xl load: %w", err)
	}

	// Step 2: compute shifted values (LAG by 1 day per ticker)
	shiftSQL := `
		CREATE OR REPLACE TABLE _xl_shift AS
		SELECT
			date, ticker,
			LAG(buy_freq, 1) OVER w AS b1,
			LAG(sell_freq, 1) OVER w AS s1,
			LAG(buy_freq + sell_freq, 1) OVER w AS t1
		FROM _xl
		WINDOW w AS (PARTITION BY ticker ORDER BY date)
	`
	if _, err := db.Exec(shiftSQL); err != nil {
		return nil, fmt.Errorf("_xl_shift: %w", err)
	}

	// Step 3: compute rolling features with WHERE filter AFTER window
	featSQL := fmt.Sprintf(`
		CREATE OR REPLACE TABLE _xl_feat AS
		WITH feat AS (
			SELECT date, ticker,
				AVG(b1) OVER (PARTITION BY ticker ORDER BY date ROWS BETWEEN 4 PRECEDING AND CURRENT ROW) AS xl_buy_freq_ma5,
				AVG(s1) OVER (PARTITION BY ticker ORDER BY date ROWS BETWEEN 4 PRECEDING AND CURRENT ROW) AS xl_sell_freq_ma5,
				AVG(b1) OVER (PARTITION BY ticker ORDER BY date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) AS xl_buy_freq_ma20,
				t1 / NULLIF(AVG(t1) OVER (PARTITION BY ticker ORDER BY date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW), 0) AS xl_freq_surge
			FROM _xl_shift
		)
		SELECT * FROM feat WHERE date = '%s'
	`, targetDate)
	if _, err := db.Exec(featSQL); err != nil {
		return nil, fmt.Errorf("_xl_feat: %w", err)
	}

	rows, err := db.Query("SELECT ticker, xl_buy_freq_ma5, xl_sell_freq_ma5, xl_buy_freq_ma20, xl_freq_surge FROM _xl_feat")
	if err != nil {
		return nil, fmt.Errorf("_xl_feat read: %w", err)
	}
	defer rows.Close()

	var result []XlRow
	for rows.Next() {
		var ticker string
		var ma5b, ma5s, ma20b, surge interface{}
		if err := rows.Scan(&ticker, &ma5b, &ma5s, &ma20b, &surge); err != nil {
			return nil, fmt.Errorf("_xl_feat scan: %w", err)
		}
		r := XlRow{Ticker: ticker, Cols: make(map[string]float64)}
		if v, ok := toFloat(ma5b); ok {
			r.Cols["xl_buy_freq_ma5"] = v
		}
		if v, ok := toFloat(ma5s); ok {
			r.Cols["xl_sell_freq_ma5"] = v
		}
		if v, ok := toFloat(ma20b); ok {
			r.Cols["xl_buy_freq_ma20"] = v
		}
		if v, ok := toFloat(surge); ok {
			r.Cols["xl_freq_surge"] = v
		}
		if len(r.Cols) > 0 {
			result = append(result, r)
		}
	}

	db.Exec("DROP TABLE IF EXISTS _xl")
	db.Exec("DROP TABLE IF EXISTS _xl_feat")
	return result, nil
}
