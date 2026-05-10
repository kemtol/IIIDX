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
			b.date,
			b.stock_code AS ticker,
			b.broker,
			m.category AS broker_category,
			TRY_CAST(m."localfund_%%" AS DOUBLE) AS localfund_pct,
			b.net_val   AS flow_total_net_buy,
			b.net_vol   AS flow_net_volume,
			b.total_val AS flow_gross_turnover,
			b.buy_freq  AS flow_buy_freq,
			b.sell_freq AS flow_sell_freq,
			ABS(b.net_val) AS flow_abs_net_buy
		FROM read_parquet('%s') b
		LEFT JOIN read_parquet('%s') m ON b.broker = m.broker_code
		WHERE b.date = '%s'
	`, broksumPath, masterBrokerPath, brokerDate)

	if _, err := db.Exec(createSQL); err != nil {
		return 0, fmt.Errorf("load L0: %w", err)
	}

	// Step 2: Context features + Bucket flags
	ctxSQL := `
		CREATE OR REPLACE TABLE _bro_ctx AS
		WITH ticker_day AS (
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
			td.ticker_day_abs_flow,
			md.market_day_abs_flow,
			CASE WHEN md.market_day_abs_flow > 0 THEN td.ticker_day_abs_flow / md.market_day_abs_flow END AS ctx_ticker_market_share,
			r.broker_net_buy_rank AS ctx_broker_net_buy_rank,
			-- Buckets
			CASE WHEN LOWER(broker_category) = 'local fund' THEN 1 ELSE 0 END AS is_localfund,
			CASE WHEN localfund_pct >= 50 THEN 1 ELSE 0 END AS is_bandar,
			CASE WHEN broker IN ('XC', 'YP', 'PD', 'SQ') THEN 1 ELSE 0 END AS is_retail
		FROM ranked r
		LEFT JOIN ticker_day td ON r.date = td.date AND r.ticker = td.ticker
		LEFT JOIN market_day md ON r.date = md.date
	`

	if _, err := db.Exec(ctxSQL); err != nil {
		return 0, fmt.Errorf("context: %w", err)
	}

	// Step 3: Global aggregate (All Brokers)
	aggSQL := `
		CREATE OR REPLACE TABLE _bro_agg AS
		SELECT 
			date, ticker,
			SUM(flow_total_net_buy) AS flow_total_net_buy_sum,
			AVG(flow_total_net_buy) AS flow_total_net_buy_mean,
			SUM(flow_abs_net_buy) AS flow_abs_net_buy_sum,
			AVG(flow_abs_net_buy) AS flow_abs_net_buy_mean,
			SUM(flow_gross_turnover) AS flow_gross_turnover_sum,
			AVG(flow_gross_turnover) AS flow_gross_turnover_mean,
			AVG(ctx_ticker_market_share) AS ctx_ticker_market_share_mean,
			AVG(ctx_broker_net_buy_rank) AS ctx_broker_net_buy_rank_mean,
			-- Retail sum
			SUM(CASE WHEN is_retail = 1 THEN flow_total_net_buy ELSE 0 END) AS total_retail_net_buy,
			-- LocalFund bucket
			SUM(CASE WHEN is_localfund = 1 THEN flow_total_net_buy ELSE 0 END) AS localfund_netbuy_sum,
			AVG(CASE WHEN is_localfund = 1 THEN flow_total_net_buy END) AS localfund_netbuy_mean,
			-- Bandar bucket
			SUM(CASE WHEN is_bandar = 1 THEN flow_total_net_buy ELSE 0 END) AS bandar_netbuy_sum,
			AVG(CASE WHEN is_bandar = 1 THEN flow_total_net_buy END) AS bandar_netbuy_mean,
			-- Ratios
			COUNT(DISTINCT broker) AS broker_count,
			SUM(CASE WHEN flow_total_net_buy > 0 THEN 1 ELSE 0 END) AS buyer_broker_count,
			SUM(CASE WHEN flow_total_net_buy < 0 THEN 1 ELSE 0 END) AS seller_broker_count
		FROM _bro_ctx
		GROUP BY date, ticker
	`
	if _, err := db.Exec(aggSQL); err != nil {
		return 0, fmt.Errorf("aggregate base: %w", err)
	}

	// Add ratio columns
	db.Exec("ALTER TABLE _bro_agg ADD COLUMN buyer_ratio DOUBLE")
	db.Exec("ALTER TABLE _bro_agg ADD COLUMN seller_ratio DOUBLE")
	db.Exec("UPDATE _bro_agg SET buyer_ratio = buyer_broker_count / NULLIF(broker_count, 0), seller_ratio = seller_broker_count / NULLIF(broker_count, 0)")


	// Step 4: Find matching columns between _bro_agg and features_store, then UPDATE
	colRows2, err := db.Query("SELECT column_name FROM information_schema.columns WHERE table_name='features_store'")
	if err != nil {
		return 0, fmt.Errorf("fs schema: %w", err)
	}
	fsCols := map[string]bool{}
	for colRows2.Next() {
		var c string
		colRows2.Scan(&c)
		fsCols[c] = true
	}
	colRows2.Close()

	// Discover all columns in _bro_agg
	colRows3, err := db.Query("SELECT name FROM pragma_table_info('_bro_agg') WHERE name NOT IN ('date', 'ticker')")
	if err != nil {
		return 0, fmt.Errorf("agg pragma: %w", err)
	}
	var setPartsSubquery []string
	var matchedCount int
	for colRows3.Next() {
		var c string
		colRows3.Scan(&c)
		if fsCols[c] {
			setPartsSubquery = append(setPartsSubquery, fmt.Sprintf(`"%s" = (SELECT "%s" FROM _bro_agg WHERE _bro_agg.ticker = features_store.ticker)`, c, c))
			matchedCount++
		}
	}
	colRows3.Close()

	if len(setPartsSubquery) == 0 {
		fmt.Fprintf(os.Stderr, "  broker: no matching columns to update\n")
		return 0, nil
	}

	updateSQL := fmt.Sprintf(`
		UPDATE features_store SET %s
		WHERE date = '%s'
		  AND EXISTS (SELECT 1 FROM _bro_agg WHERE _bro_agg.ticker = features_store.ticker)
	`, strings.Join(setPartsSubquery, ", "), targetDate)
	fmt.Fprintf(os.Stderr, "  broker cols matched=%d\n", matchedCount)

	res, err := db.Exec(updateSQL)
	if err != nil {
		return 0, fmt.Errorf("update: %w", err)
	}
	n, _ := res.RowsAffected()

	// Verify update actually worked
	var verifyCount int
	db.QueryRow("SELECT COUNT(*) FROM features_store WHERE date = ? AND flow_total_net_buy_sum IS NOT NULL", targetDate).Scan(&verifyCount)
	fmt.Fprintf(os.Stderr, "  broker debug: RowsAffected=%d, verify non-null=%d\n", n, verifyCount)
	
	db.Exec("DROP TABLE IF EXISTS _bro")
	db.Exec("DROP TABLE IF EXISTS _bro_ctx")
	db.Exec("DROP TABLE IF EXISTS _bro_agg")
	return verifyCount, nil
}
