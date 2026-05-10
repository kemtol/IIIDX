package features

import (
	"database/sql"
	"encoding/csv"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"

	_ "github.com/marcboeker/go-duckdb"
)

func TestFeatureParity(t *testing.T) {
	// 1. Setup temporary DuckDB
	dbPath := "parity_test.duckdb"
	os.Remove(dbPath)
	db, err := sql.Open("duckdb", dbPath)
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	defer os.Remove(dbPath)

	// 2. Load Gold Standard CSV
	goldPath := "gold_standard_20260506.csv"
	f, err := os.Open(goldPath)
	if err != nil {
		t.Skip("Gold standard CSV not found, skipping parity test")
	}
	defer f.Close()

	reader := csv.NewReader(f)
	records, err := reader.ReadAll()
	if err != nil {
		t.Fatal(err)
	}
	header := records[0]
	
	// Map of Ticker -> Column -> Value
	goldData := make(map[string]map[string]float64)
	for i := 1; i < len(records); i++ {
		row := records[i]
		ticker := ""
		tickerIdx := -1
		for idx, h := range header {
			if h == "ticker" {
				ticker = row[idx]
				tickerIdx = idx
				break
			}
		}
		if ticker == "" { continue }
		
		goldData[ticker] = make(map[string]float64)
		for idx, h := range header {
			if idx == tickerIdx || h == "date" { continue }
			val, _ := strconv.ParseFloat(row[idx], 64)
			goldData[ticker][h] = val
		}
	}

	// 3. Mock/Load required L0 data
	// Note: For this test to be truly independent, we'd need to mock the Parquet files.
	// For now, we assume the local data/Level_0_Raw is available.
	repoRoot := "../../../../.."
	targetDate := "2026-05-06"
	
	// Create features_store table
	db.Exec(`CREATE TABLE features_store (date DATE, ticker VARCHAR, entry_price DOUBLE)`)
	db.Exec(`CREATE UNIQUE INDEX idx_features_date_ticker ON features_store (date, ticker)`)

	// 4. Run Go implementations (Partial subset for P0)
	// We'll add more as we implement parity.
	t.Run("BrokerParity", func(t *testing.T) {
		brokerDate := "2026-05-04" // Mocking logic: T-1/T-2
		bksPath := filepath.Join(repoRoot, "data/Level_0_Raw/broksum_bybroker.parquet")
		mbPath := filepath.Join(repoRoot, "data/Level_0_Raw/master_broker.parquet")
		
		// Ensure columns exist (simulating main.go ALTER logic)
		for col := range goldData["ABDA"] {
			if strings.HasPrefix(col, "flow_") || strings.HasPrefix(col, "ctx_") {
				db.Exec(fmt.Sprintf(`ALTER TABLE features_store ADD COLUMN IF NOT EXISTS "%s" DOUBLE`, col))
			}
		}
		
		// We need to INSERT tickers into features_store first so UPDATE works
		for ticker := range goldData {
			db.Exec("INSERT INTO features_store (date, ticker) VALUES (?, ?)", targetDate, ticker)
		}

		_, err := ComputeBrokerAggregate(db, bksPath, mbPath, brokerDate, targetDate)
		if err != nil {
			t.Fatal(err)
		}

		// Check parity for a few sample columns
		checkCols := []string{"flow_total_net_buy_sum", "flow_abs_net_buy_mean", "ctx_broker_ticker_specificity_mean"}
		for ticker, goldFeatures := range goldData {
			for _, col := range checkCols {
				var got sql.NullFloat64
				err := db.QueryRow(fmt.Sprintf(`SELECT "%s" FROM features_store WHERE ticker = ?`, col), ticker).Scan(&got)
				if err != nil {
					t.Errorf("%s %s: query error %v", ticker, col, err)
					continue
				}
				
				want := goldFeatures[col]
				if !got.Valid {
					t.Errorf("%s %s: got NULL, want %f", ticker, col, want)
					continue
				}
				
				if math.Abs(got.Float64 - want) > 1e-5 {
					t.Errorf("%s %s: got %f, want %f (diff %f)", ticker, col, got.Float64, want, math.Abs(got.Float64 - want))
				}
			}
		}
	})
}
