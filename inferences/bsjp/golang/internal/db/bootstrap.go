package db

import (
	"database/sql"
	"fmt"
)

func Bootstrap(db *sql.DB, l2Path string) (int64, error) {
	db.Exec("DROP TABLE IF EXISTS features_store")

	_, err := db.Exec(fmt.Sprintf(`
		CREATE TABLE features_store AS
		SELECT *, now() as inserted_at
		FROM read_parquet('%s')
	`, l2Path))
	if err != nil {
		return 0, fmt.Errorf("bootstrap create: %w", err)
	}

	// Add PK to allow UPSERT
	_, err = db.Exec("CREATE UNIQUE INDEX IF NOT EXISTS idx_features_date_ticker ON features_store (date, ticker)")
	if err != nil {
		return 0, fmt.Errorf("bootstrap pk: %w", err)
	}

	// Ensure picks_log exists
	db.Exec(`CREATE TABLE IF NOT EXISTS picks_log (
		date DATE, variant VARCHAR, rank INTEGER, ticker VARCHAR,
		pred_proba DOUBLE, entry_price DOUBLE, exit_price DOUBLE,
		overnight_return DOUBLE, hit_tp BOOLEAN, logged_at TIMESTAMP DEFAULT now(),
		PRIMARY KEY (date, variant, rank))`)

	var count int64
	db.QueryRow("SELECT COUNT(*) FROM features_store").Scan(&count)
	return count, nil
}
