package db

import (
	"database/sql"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"
	"time"

	_ "github.com/marcboeker/go-duckdb"
)

func Open(path string) (*sql.DB, error) {
	if err := os.MkdirAll(filepath.Dir(path), 0755); err != nil {
		return nil, err
	}
	db, err := sql.Open("duckdb", path)
	if err != nil {
		return nil, err
	}
	db.SetMaxOpenConns(1)
	db.Exec("PRAGMA memory_limit='512MB'")
	db.Exec("PRAGMA threads=2")
	return db, nil
}

// FeatureRow holds a map of column → value for one (date, ticker).
type FeatureRow struct {
	Ticker string
	Cols   map[string]*float64
}

func NewFeatureRow(ticker string) FeatureRow {
	return FeatureRow{Ticker: ticker, Cols: make(map[string]*float64)}
}

func (r *FeatureRow) Set(col string, v float64) {
	if math.IsNaN(v) || math.IsInf(v, 0) {
		return
	}
	r.Cols[col] = &v
}

func (r *FeatureRow) SetPtr(col string, v *float64) {
	if v == nil {
		return
	}
	r.Cols[col] = v
}

// UpsertDate upserts features for a single date.
func UpsertDate(db *sql.DB, date string, rows []FeatureRow) (int, error) {
	if len(rows) == 0 {
		return 0, nil
	}

	// DuckDB segmentation fault prevention:
	// Use a single multi-row INSERT or smaller batches to minimize CGO roundtrips.
	if _, err := db.Exec("DELETE FROM features_store WHERE date = ?", date); err != nil {
		return 0, fmt.Errorf("delete existing rows for %s: %w", date, err)
	}

	colNames := make([]string, 0, len(rows[0].Cols))
	for c := range rows[0].Cols {
		colNames = append(colNames, c)
	}
	colList := strings.Join(colNames, ", ")

	const batchSize = 100
	for i := 0; i < len(rows); i += batchSize {
		end := i + batchSize
		if end > len(rows) {
			end = len(rows)
		}

		batch := rows[i:end]
		var valueStrings []string
		for _, r := range batch {
			vals := make([]string, 0, 2+len(colNames))
			vals = append(vals, fmt.Sprintf("'%s'", date))
			vals = append(vals, fmt.Sprintf("'%s'", strings.ReplaceAll(r.Ticker, "'", "''")))

			for _, c := range colNames {
				v := r.Cols[c]
				if v == nil {
					vals = append(vals, "NULL")
				} else {
					// Use %g to avoid excessive zeros and handle large/small numbers efficiently
					vals = append(vals, fmt.Sprintf("%g", *v))
				}
			}
			valueStrings = append(valueStrings, "("+strings.Join(vals, ", ")+")")
		}

		sql := fmt.Sprintf("INSERT OR REPLACE INTO features_store (date, ticker, %s) VALUES %s",
			colList, strings.Join(valueStrings, ", "))

		if _, err := db.Exec(sql); err != nil {
			return i, fmt.Errorf("upsert batch %d-%d: %w", i, end, err)
		}
	}

	return len(rows), nil
}
func GetLatestDate(db *sql.DB) (string, error) {
	var latest *time.Time
	err := db.QueryRow("SELECT MAX(date) FROM features_store").Scan(&latest)
	if err != nil || latest == nil {
		return "", nil
	}
	return latest.Format("2006-01-02"), nil
}

func LogPicks(db *sql.DB, picks []PickRow) error {
	db.Exec(`CREATE TABLE IF NOT EXISTS picks_log (
		date DATE, variant VARCHAR, rank INTEGER, ticker VARCHAR,
		pred_proba DOUBLE, entry_price DOUBLE, exit_price DOUBLE,
		overnight_return DOUBLE, hit_tp BOOLEAN, logged_at TIMESTAMP DEFAULT now(),
		PRIMARY KEY (date, variant, rank))`)

	stmt, err := db.Prepare(
		"INSERT INTO picks_log (date, variant, rank, ticker, pred_proba, entry_price) " +
			"VALUES (?, ?, ?, ?, ?, ?) " +
			"ON CONFLICT (date, variant, rank) DO UPDATE SET " +
			"ticker=excluded.ticker, pred_proba=excluded.pred_proba, " +
			"entry_price=excluded.entry_price, logged_at=now()")
	if err != nil {
		return err
	}
	defer stmt.Close()
	for _, p := range picks {
		if _, err := stmt.Exec(p.Date, p.Variant, p.Rank, p.Ticker, p.PredProba, p.EntryPrice); err != nil {
			return err
		}
	}
	return nil
}

type PickRow struct {
	Date       string
	Variant    string
	Rank       int
	Ticker     string
	PredProba  *float64
	EntryPrice *float64
}

func Check(db *sql.DB) error {
	var count int64
	var minDate, maxDate *time.Time
	db.QueryRow("SELECT COUNT(*) FROM features_store").Scan(&count)
	db.QueryRow("SELECT MIN(date) FROM features_store").Scan(&minDate)
	db.QueryRow("SELECT MAX(date) FROM features_store").Scan(&maxDate)
	if maxDate != nil && minDate != nil {
		fmt.Printf("features_store: %d rows, %s → %s\n", count,
			minDate.Format("2006-01-02"), maxDate.Format("2006-01-02"))
	} else {
		fmt.Println("features_store: empty. Run bootstrap or fetch first.")
	}
	return nil
}
