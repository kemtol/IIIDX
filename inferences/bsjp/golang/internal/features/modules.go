package features

import (
	"database/sql"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

// seedFromMomentum creates base (date, ticker) rows in features_store from
// the closing_momentum module. Returns the count of rows inserted.
func seedFromMomentum(db *sql.DB, modulesDir, targetDate string) (int, error) {
	momPath := filepath.Join(modulesDir, "closing_momentum_features.parquet")
	if _, err := os.Stat(momPath); os.IsNotExist(err) {
		return 0, fmt.Errorf("closing_momentum_features.parquet not found")
	}

	// Load momentum data for latest date <= targetDate, overriding date to targetDate
	loadSQL := fmt.Sprintf(`
		CREATE OR REPLACE TEMP TABLE _mom_seed AS
		SELECT '%s' AS date, ticker, close_ret_last1h, close_vs_open_day, close_range_pct
		FROM read_parquet('%s')
		WHERE date = (SELECT MAX(date) FROM read_parquet('%s') WHERE date <= '%s')
	`, targetDate, momPath, momPath, targetDate)

	if _, err := db.Exec(loadSQL); err != nil {
		return 0, err
	}

	// Ensure columns exist in features_store
	for _, c := range []string{"close_ret_last1h", "close_vs_open_day", "close_range_pct"} {
		db.Exec(fmt.Sprintf(`ALTER TABLE features_store ADD COLUMN IF NOT EXISTS "%s" DOUBLE`, c))
	}

	// Insert by name (only matching columns)
	res, err := db.Exec("INSERT INTO features_store BY NAME SELECT * FROM _mom_seed")
	if err != nil {
		return 0, err
	}
	n, _ := res.RowsAffected()
	return int(n), nil
}

// UpsertModules reads all *_features.parquet from modulesDir for the latest
// date <= targetDate, then upserts each module's columns into features_store
// via individual UPDATE statements. Handles per-ticker and per-date modules.
func UpsertModules(db *sql.DB, modulesDir, targetDate string) (int, error) {
	matches, _ := filepath.Glob(filepath.Join(modulesDir, "*_features.parquet"))
	if len(matches) == 0 {
		return 0, fmt.Errorf("no module files in %s", modulesDir)
	}

	// Ensure features_store has rows for targetDate.
	// If not (new date not in bootstrap), seed from closing_momentum module.
	baseCount := 0
	db.QueryRow("SELECT COUNT(*) FROM features_store WHERE date = ?", targetDate).Scan(&baseCount)
	if baseCount == 0 {
		cmp, err := seedFromMomentum(db, modulesDir, targetDate)
		if err != nil {
			return 0, fmt.Errorf("seed momentum: %w", err)
		}
		if cmp == 0 {
			return 0, fmt.Errorf("no features_store rows for %s and could not seed from momentum", targetDate)
		}
		fmt.Printf("  seeded %d ticker rows from closing_momentum for %s\n", cmp, targetDate)
	}

	totalUpdated := 0

	for _, path := range matches {
		name := filepath.Base(path)
		if !strings.HasSuffix(name, "_features.parquet") {
			continue
		}

		// Create temp table with latest date <= targetDate
		createSQL := fmt.Sprintf(`
			CREATE OR REPLACE TEMP TABLE _mod AS
			SELECT * FROM read_parquet('%s')
			WHERE date = (SELECT MAX(date) FROM read_parquet('%s') WHERE date <= '%s')
		`, path, path, targetDate)

		if _, err := db.Exec(createSQL); err != nil {
			fmt.Fprintf(os.Stderr, "  skip %s: create temp: %v\n", name, err)
			continue
		}

		// Discover columns from _mod
		colRows, err := db.Query("SELECT name FROM pragma_table_info('_mod') WHERE name NOT IN ('date', 'ticker')")
		if err != nil {
			fmt.Fprintf(os.Stderr, "  skip %s: pragma: %v\n", name, err)
			continue
		}

		var modCols []string
		for colRows.Next() {
			var c string
			colRows.Scan(&c)
			modCols = append(modCols, c)
		}
		colRows.Close()

		if len(modCols) == 0 {
			continue
		}

		// Ensure features_store has all these columns
		for _, c := range modCols {
			alterSQL := fmt.Sprintf(`ALTER TABLE features_store ADD COLUMN IF NOT EXISTS "%s" DOUBLE`, c)
			db.Exec(alterSQL) // ignore errors (column may exist)
		}

		// Build SET clause
		var setClauses []string
		for _, c := range modCols {
			setClauses = append(setClauses, fmt.Sprintf(`"%s" = _mod."%s"`, c, c))
		}

		// Check if module has ticker column
		var hasTicker int
		db.QueryRow("SELECT COUNT(*) FROM pragma_table_info('_mod') WHERE name = 'ticker'").Scan(&hasTicker)

		var updateSQL string
		if hasTicker > 0 {
			updateSQL = fmt.Sprintf(`
				UPDATE features_store SET %s
				FROM _mod
				WHERE features_store.date = '%s'
				  AND features_store.ticker = _mod.ticker
			`, strings.Join(setClauses, ", "), targetDate)
		} else {
			// Per-date module (e.g. global_indices): broadcast to all tickers for targetDate
			updateSQL = fmt.Sprintf(`
				UPDATE features_store SET %s
				FROM _mod
				WHERE features_store.date = '%s'
			`, strings.Join(setClauses, ", "), targetDate)
		}

		res, err := db.Exec(updateSQL)
		if err != nil {
			fmt.Fprintf(os.Stderr, "  skip %s: update: %v\n", name, err)
			continue
		}
		n, _ := res.RowsAffected()
		totalUpdated += int(n)
		fmt.Printf("  %s: %d rows updated\n", name, n)
	}

	return totalUpdated, nil
}
