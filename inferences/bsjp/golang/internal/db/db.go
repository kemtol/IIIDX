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

// UpsertDate upserts features for a single date. Only inserts columns present in the rows.
func UpsertDate(db *sql.DB, date string, rows []FeatureRow) (int, error) {
	if len(rows) == 0 {
		return 0, nil
	}

	// Collect all column names from the first row
	colNames := make([]string, 0, len(rows[0].Cols))
	for c := range rows[0].Cols {
		colNames = append(colNames, c)
	}

	tx, err := db.Begin()
	if err != nil {
		return 0, err
	}
	defer tx.Rollback()

	colList := strings.Join(colNames, ", ")
	placeholders := strings.Repeat("?,", len(colNames))
	placeholders = placeholders[:len(placeholders)-1]

	sql := fmt.Sprintf("INSERT OR REPLACE INTO features_store (date, ticker, %s) VALUES (?, ?, %s)", colList, placeholders)
	stmt, err := tx.Prepare(sql)
	if err != nil {
		return 0, err
	}
	defer stmt.Close()

	for _, r := range rows {
		args := make([]interface{}, 2+len(colNames))
		args[0] = date
		args[1] = r.Ticker
		for i, c := range colNames {
			args[2+i] = r.Cols[c]
		}
		if _, err := stmt.Exec(args...); err != nil {
			return 0, fmt.Errorf("insert %s: %w", r.Ticker, err)
		}
	}
	if err := tx.Commit(); err != nil {
		return 0, err
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
