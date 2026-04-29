package features

import (
	"database/sql"
	"fmt"
	"os"
	"strings"
)

// ComputeBrokerAggregate reads broksum_bybroker.parquet (L0, grain: date/broker/stock_code),
// computes base + context broker features, aggregates to (date, ticker), and upserts to features_store.
func ComputeBrokerAggregate(db *sql.DB, broksumPath, masterBrokerPath, brokerDate, targetDate string) (int, error) {
	if _, err := os.Stat(broksumPath); os.IsNotExist(err) {
		return 0, fmt.Errorf("broksum_bybroker.parquet not found: %s", broksumPath)
	}

	// Step 1: Load raw L0 data with column renaming + base feature computation
	createSQL := fmt.Sprintf(`
		CREATE OR REPLACE TABLE _bro AS
		SELECT
			date,
			stock_code AS ticker,
			broker,
			net_val   AS flow_total_net_buy,
			buy_val   AS flow_gross_buy,
			sell_val  AS flow_gross_sell,
			total_val AS flow_gross_turnover,
			buy_vol   AS flow_buy_volume,
			sell_vol  AS flow_sell_volume,
			net_vol   AS flow_net_volume,
			buy_freq  AS flow_buy_freq,
			sell_freq AS flow_sell_freq,
			avg_buy_price  AS flow_avg_buy_price,
			avg_sell_price AS flow_avg_sell_price,

			-- base features
			ABS(net_val) AS flow_abs_net_buy,
			CASE WHEN total_val > 0 THEN net_val / total_val END AS flow_net_flow_ratio,
			CASE WHEN sell_val > 0 AND (buy_val <= 0 OR sell_val > 0) THEN
				CASE WHEN buy_val = 0 THEN 99.0
				     WHEN sell_val = 0 THEN 0.01
				     ELSE buy_val / sell_val END
			END AS flow_buy_sell_val_ratio,
			CASE WHEN sell_vol > 0 THEN
				CASE WHEN buy_vol = 0 THEN 99.0
				     WHEN sell_vol = 0 THEN 0.01
				     ELSE buy_vol / sell_vol END
			END AS flow_buy_sell_vol_ratio,
			CASE WHEN sell_freq > 0 THEN
				CASE WHEN buy_freq = 0 THEN 99.0
				     WHEN sell_freq = 0 THEN 0.01
				     ELSE buy_freq / sell_freq END
			END AS flow_buy_sell_freq_ratio,
			buy_freq + sell_freq AS flow_total_trades,
			CASE WHEN (buy_freq + sell_freq) > 0 THEN net_val / (buy_freq + sell_freq) END AS flow_net_buy_per_trade,
			CASE WHEN ABS(net_val) > 0 THEN total_val / ABS(net_val) END AS flow_churn_ratio,
			avg_sell_price - avg_buy_price AS flow_avg_spread_price,
			CASE WHEN avg_buy_price > 0 THEN (avg_sell_price - avg_buy_price) / avg_buy_price END AS flow_avg_spread_pct
		FROM read_parquet('%s')
		WHERE date = '%s'
	`, broksumPath, brokerDate)

	if _, err := db.Exec(createSQL); err != nil {
		return 0, fmt.Errorf("load L0: %w", err)
	}

	var l0Count int
	db.QueryRow("SELECT COUNT(*) FROM _bro").Scan(&l0Count)
	if l0Count == 0 {
		fmt.Fprintf(os.Stderr, "  broker: no L0 rows for %s\n", targetDate)
		return 0, nil
	}

	// Step 2: Context features (per date/broker, date/ticker, and market aggregates)
	ctxSQL := `
		CREATE OR REPLACE TABLE _bro_ctx AS
		WITH broker_day AS (
			SELECT date, broker, SUM(flow_abs_net_buy) AS broker_day_abs_flow
			FROM _bro GROUP BY date, broker
		),
		ticker_day AS (
			SELECT date, ticker, SUM(flow_abs_net_buy) AS ticker_day_abs_flow
			FROM _bro GROUP BY date, ticker
		),
		market_day AS (
			SELECT date, SUM(flow_abs_net_buy) AS market_day_abs_flow
			FROM _bro GROUP BY date
		),
		ranked AS (
			SELECT *,
				PERCENT_RANK() OVER (PARTITION BY date, ticker ORDER BY flow_total_net_buy) AS broker_net_buy_rank
			FROM _bro
		)
		SELECT
			r.*,
			bd.broker_day_abs_flow,
			td.ticker_day_abs_flow,
			md.market_day_abs_flow,
			CASE WHEN bd.broker_day_abs_flow > 0 THEN flow_abs_net_buy / bd.broker_day_abs_flow END AS ctx_broker_ticker_specificity,
			CASE WHEN md.market_day_abs_flow > 0 THEN bd.broker_day_abs_flow / md.market_day_abs_flow END AS ctx_broker_market_share,
			CASE WHEN md.market_day_abs_flow > 0 THEN td.ticker_day_abs_flow / md.market_day_abs_flow END AS ctx_ticker_market_share,
			r.broker_net_buy_rank AS ctx_broker_net_buy_rank
		FROM ranked r
		LEFT JOIN broker_day bd ON r.date = bd.date AND r.broker = bd.broker
		LEFT JOIN ticker_day td ON r.date = td.date AND r.ticker = td.ticker
		LEFT JOIN market_day md ON r.date = md.date
	`

	if _, err := db.Exec(ctxSQL); err != nil {
		return 0, fmt.Errorf("context: %w", err)
	}

	// Step 3: Aggregate to (date, ticker) grain
	// Dynamically discover numeric columns from _bro_ctx
	colRows, err := db.Query("SELECT name FROM pragma_table_info('_bro_ctx') WHERE name NOT IN ('date','ticker','broker','flow_gross_buy','flow_gross_sell') AND type IN ('DOUBLE','FLOAT','BIGINT','INTEGER','HUGEINT','SMALLINT','TINYINT','DECIMAL','BIGNUM','REAL')")
	if err != nil {
		return 0, fmt.Errorf("pragma: %w", err)
	}
	var numCols []string
	for colRows.Next() {
		var c string
		colRows.Scan(&c)
		numCols = append(numCols, c)
	}
	colRows.Close()

	if len(numCols) == 0 {
		return 0, nil
	}

	var selectParts []string
	for _, c := range numCols {
		selectParts = append(selectParts,
			fmt.Sprintf(`SUM(%s) AS %s_sum`, c, c),
			fmt.Sprintf(`AVG(%s) AS %s_mean`, c, c),
		)
	}

	aggSQL := fmt.Sprintf(`
		CREATE OR REPLACE TABLE _bro_agg AS
		SELECT date, ticker, %s
		FROM _bro_ctx
		GROUP BY date, ticker
	`, strings.Join(selectParts, ", "))

	if _, err := db.Exec(aggSQL); err != nil {
		return 0, fmt.Errorf("aggregate: %w", err)
	}

	// Step 4: Find matching columns between _bro_agg and features_store, then UPDATE
	colRows2, err := db.Query("SELECT column_name FROM information_schema.columns WHERE table_name='features_store'")
	if err != nil {
		return 0, fmt.Errorf("fs schema: %w", err)
	}
	fsCols := map[string]bool{}
	var list []string
	for colRows2.Next() {
		var c string
		colRows2.Scan(&c)
		fsCols[c] = true
		list = append(list, c)
	}
	colRows2.Close()
	_ = list

	var setParts []string
	var setPartsSubquery []string
	for _, c := range numCols {
		for _, suffix := range []string{"_sum", "_mean"} {
			cn := c + suffix
			if fsCols[cn] {
				setParts = append(setParts, cn)
				setPartsSubquery = append(setPartsSubquery, fmt.Sprintf(`"%s" = (SELECT "%s" FROM _bro_agg WHERE _bro_agg.ticker = features_store.ticker)`, cn, cn))
			}
		}
	}

	if len(setParts) == 0 {
		fmt.Fprintf(os.Stderr, "  broker: no matching columns to update\n")
		return 0, nil
	}

	updateSQL := fmt.Sprintf(`
		UPDATE features_store SET %s
		WHERE date = '%s'
		  AND EXISTS (SELECT 1 FROM _bro_agg WHERE _bro_agg.ticker = features_store.ticker)
	`, strings.Join(setPartsSubquery, ", "), targetDate)
	fmt.Fprintf(os.Stderr, "  broker cols=%d setParts=%d\n", len(setParts), len(setPartsSubquery))
	var aggCount int
	db.QueryRow("SELECT COUNT(*) FROM _bro_agg").Scan(&aggCount)
	fmt.Fprintf(os.Stderr, "  broker debug: _bro_agg=%d rows, fs cols matched=%d\n", aggCount, len(setParts))

	res, err := db.Exec(updateSQL)
	if err != nil {
		return 0, fmt.Errorf("update: %w", err)
	}
	n, _ := res.RowsAffected()

	// Verify update actually worked
	var verifyCount int
	db.QueryRow("SELECT COUNT(*) FROM features_store WHERE date = ? AND flow_total_net_buy_sum IS NOT NULL", targetDate).Scan(&verifyCount)
	fmt.Fprintf(os.Stderr, "  broker debug: RowsAffected=%d, verify non-null=%d\n", n, verifyCount)
	fmt.Printf("  broker (from L0): %d rows updated from %d L0 rows (%d cols)\n", verifyCount, l0Count, len(setParts))
	db.Exec("DROP TABLE IF EXISTS _bro")
	db.Exec("DROP TABLE IF EXISTS _bro_ctx")
	db.Exec("DROP TABLE IF EXISTS _bro_agg")
	return verifyCount, nil
}
