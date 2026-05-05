package features

import (
	"database/sql"
	"fmt"
	"os"
	"strings"
)

func ComputeCVD(db *sql.DB, broksumPath, brokerDate, targetDate string) (int, error) {
	if _, err := os.Stat(broksumPath); os.IsNotExist(err) {
		return 0, fmt.Errorf("broksum_bybroker.parquet not found: %s", broksumPath)
	}

	createRaw := fmt.Sprintf(`
		CREATE OR REPLACE TABLE _cvd_raw AS
		SELECT date, stock_code AS ticker,
			SUM(net_vol) AS net_volume,
			SUM(buy_vol) AS buy_volume,
			SUM(sell_vol) AS sell_volume
		FROM read_parquet('%s')
		WHERE date <= '%s'
		GROUP BY date, stock_code
	`, broksumPath, brokerDate)

	if _, err := db.Exec(createRaw); err != nil {
		return 0, fmt.Errorf("cvd raw: %w", err)
	}

	var rawCount int
	db.QueryRow("SELECT COUNT(*) FROM _cvd_raw").Scan(&rawCount)
	if rawCount == 0 {
		fmt.Fprintf(os.Stderr, "  cvd: no L0 rows through %s\n", brokerDate)
		return 0, nil
	}

	cvdSQL := fmt.Sprintf(`
		CREATE OR REPLACE TABLE _cvd AS
		WITH ticker_sorted AS (
			SELECT date, ticker, net_volume,
				(buy_volume + sell_volume) AS total_volume
			FROM _cvd_raw
		),
		with_rolling AS (
			SELECT date, ticker,
				SUM(net_volume) OVER w5 AS cvd_5d,
				SUM(net_volume) OVER w10 AS cvd_10d,
				SUM(net_volume) OVER w20 AS cvd_20d,
				CASE WHEN SUM(total_volume) OVER w5 > 0
					THEN SUM(net_volume) OVER w5 / SUM(total_volume) OVER w5
				END AS cvd_5d_norm,
				CASE WHEN SUM(total_volume) OVER w10 > 0
					THEN SUM(net_volume) OVER w10 / SUM(total_volume) OVER w10
				END AS cvd_10d_norm,
				CASE WHEN SUM(total_volume) OVER w20 > 0
					THEN SUM(net_volume) OVER w20 / SUM(total_volume) OVER w20
				END AS cvd_20d_norm
			FROM ticker_sorted
			WINDOW
				w5  AS (PARTITION BY ticker ORDER BY date ROWS BETWEEN 4 PRECEDING AND CURRENT ROW),
				w10 AS (PARTITION BY ticker ORDER BY date ROWS BETWEEN 9 PRECEDING AND CURRENT ROW),
				w20 AS (PARTITION BY ticker ORDER BY date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW)
		)
		SELECT * FROM with_rolling
		WHERE date = '%s'
	`, brokerDate)

	if _, err := db.Exec(cvdSQL); err != nil {
		return 0, fmt.Errorf("cvd compute: %w", err)
	}

	var cvdCount int
	db.QueryRow("SELECT COUNT(*) FROM _cvd").Scan(&cvdCount)
	if cvdCount == 0 {
		fmt.Fprintf(os.Stderr, "  cvd: no rows for %s\n", brokerDate)
		db.Exec("DROP TABLE IF EXISTS _cvd_raw")
		return 0, nil
	}

	cvdCols := []string{"cvd_5d", "cvd_5d_norm", "cvd_10d", "cvd_10d_norm", "cvd_20d", "cvd_20d_norm"}
	for _, c := range cvdCols {
		db.Exec(fmt.Sprintf(`ALTER TABLE features_store ADD COLUMN IF NOT EXISTS "%s" DOUBLE`, c))
	}

	var setParts []string
	for _, c := range cvdCols {
		setParts = append(setParts, fmt.Sprintf(`"%s" = (SELECT "%s" FROM _cvd WHERE _cvd.ticker = features_store.ticker)`, c, c))
	}

	updateSQL := fmt.Sprintf(`
		UPDATE features_store SET %s
		WHERE date = '%s'
			AND EXISTS (SELECT 1 FROM _cvd WHERE _cvd.ticker = features_store.ticker)
	`, strings.Join(setParts, ", "), targetDate)

	res, err := db.Exec(updateSQL)
	if err != nil {
		return 0, fmt.Errorf("cvd update: %w", err)
	}
	n, _ := res.RowsAffected()

	var verifyCount int
	db.QueryRow("SELECT COUNT(*) FROM features_store WHERE date = ? AND cvd_20d_norm IS NOT NULL", targetDate).Scan(&verifyCount)
	fmt.Printf("  cvd (from L0): %d rows updated (RowsAffected=%d)\n", verifyCount, n)

	db.Exec("DROP TABLE IF EXISTS _cvd_raw")
	db.Exec("DROP TABLE IF EXISTS _cvd")
	return verifyCount, nil
}
