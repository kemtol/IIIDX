package main

import (
	"bufio"
	"database/sql"
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"os"
	"sort"
	"strings"
	"time"

	"github.com/mmmachine/bsjp/internal/config"
	"github.com/mmmachine/bsjp/internal/db"
	"github.com/mmmachine/bsjp/internal/features"
	"github.com/mmmachine/bsjp/internal/fetcher"
	"github.com/mmmachine/bsjp/internal/model"
	"github.com/mmmachine/bsjp/internal/notify"
	"github.com/parquet-go/parquet-go"
)

func main() {
	if err := run(); err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}
}

func run() error {
	if len(os.Args) < 2 {
		fmt.Println("bsjp — BSJP inference engine")
		fmt.Println("\nSubcommands:")
		fmt.Println("  bootstrap   Backfill DuckDB from L2 parquet")
		fmt.Println("  fetch       Compute features from L0 and upsert")
		fmt.Println("  predict     Predict top-3 picks")
		fmt.Println("  download    Fetch 1h bars from Yahoo Finance")
		fmt.Println("  check       Print freshness report")
		return nil
	}

	// Parse repo-root from os.Args manually (before subcommand)
	var repoRoot string
	for i, a := range os.Args {
		if a == "--repo-root" && i+1 < len(os.Args) {
			repoRoot = os.Args[i+1]
			break
		}
	}

	switch os.Args[1] {
	case "bootstrap":
		return cmdBootstrap(repoRoot, os.Args[2:])
	case "fetch":
		return cmdFetch(repoRoot, os.Args[2:])
	case "predict":
		return cmdPredict(repoRoot, os.Args[2:])
	case "download":
		return cmdDownload(repoRoot, os.Args[2:])
	case "check":
		return cmdCheck(repoRoot, os.Args[2:])
	default:
		fmt.Fprintf(os.Stderr, "Unknown subcommand: %s\n", os.Args[1])
		os.Exit(1)
	}
	return nil
}

func loadConfig(repoRoot string) (*config.Config, error) {
	if repoRoot != "" {
		return config.FromRoot(repoRoot), nil
	}
	return config.FromCWD()
}

// ── bootstrap ─────────────────────────────────────────────────────────────

func cmdBootstrap(repoRoot string, args []string) error {
	f := flag.NewFlagSet("bootstrap", flag.ExitOnError)
	l2Parquet := f.String("l2-parquet", "", "Path to L2 parquet")
	dbPath := f.String("db", "", "Path to DuckDB")
	f.Parse(args)

	cfg, err := loadConfig(repoRoot)
	if err != nil {
		return err
	}

	l2 := cfg.L2Parquet
	if *l2Parquet != "" {
		l2 = *l2Parquet
	}
	duckPath := cfg.DuckDBPath
	if *dbPath != "" {
		duckPath = *dbPath
	}

	if _, err := os.Stat(l2); os.IsNotExist(err) {
		return fmt.Errorf("L2 parquet not found at %s", l2)
	}

	fmt.Printf("Bootstrapping from %s\n", l2)
	database, err := db.Open(duckPath)
	if err != nil {
		return err
	}
	defer database.Close()

	count, err := db.Bootstrap(database, l2)
	if err != nil {
		return err
	}
	fmt.Printf("Done: %d rows inserted\n", count)
	return nil
}

// ── fetch ─────────────────────────────────────────────────────────────────

func cmdFetch(repoRoot string, args []string) error {
	f := flag.NewFlagSet("fetch", flag.ExitOnError)
	date := f.String("date", "today", "Target date (YYYY-MM-DD or 'today')")
	force := f.Bool("force", false, "Force recompute")
	dbPath := f.String("db", "", "Path to DuckDB (default: config)")
	f.Parse(args)

	// Resolve target date
	targetDate := *date
	if targetDate == "today" {
		targetDate = time.Now().Format("2006-01-02")
	}

	cfg, err := loadConfig(repoRoot)
	if err != nil {
		return err
	}

	duckPath := cfg.DuckDBPath
	if *dbPath != "" {
		duckPath = *dbPath
	}

	database, err := db.Open(duckPath)
	if err != nil {
		return err
	}
	defer database.Close()

	if !*force {
		latest, _ := db.GetLatestDate(database)
		if latest >= targetDate {
			fmt.Printf("Already up to date (latest: %s)\n", latest)
			return nil
		}
	}

	// Compute features from L0 data
	yf1h := cfg.L0File("yfinance_1h.parquet")
	if _, err := os.Stat(yf1h); os.IsNotExist(err) {
		return fmt.Errorf("yfinance_1h.parquet not found at %s", yf1h)
	}
	fmt.Printf("Computing features for %s...\n", targetDate)

	// ── Step 2: Seed & Load Modular Features (Hybrid Path) ──
	// This seeds base ticker rows and fills historical/complex features from research parquets
	nMod, err := features.UpsertModules(database, cfg.ModulesDir, targetDate)
	if err != nil {
		return fmt.Errorf("upsert modules: %w", err)
	}
	fmt.Printf("  modular features: %d updates from research parquets\n", nMod)

	// ── Step 3: Real-time Features (L0 Path) ──
	// We override/fill the most recent data directly from L0 parquets
	nMom, err := upsertLiveMomentum(database, yf1h, targetDate)
	if err != nil {
		return fmt.Errorf("live momentum: %w", err)
	}
	fmt.Printf("  live momentum/entry: %d rows updated from yfinance_1h\n", nMom)

	nP14, err := upsertLivePreclose14(database, yf1h, targetDate)
	if err != nil {
		return fmt.Errorf("live preclose14: %w", err)
	}
	fmt.Printf("  live preclose14: %d rows updated from yfinance_1h\n", nP14)

	bksPath := cfg.L0File("broksum_bybroker.parquet")
	mbPath := cfg.L0File("master_broker.parquet")

	// Find most recent broker date < targetDate (T-1 safety)
	brokerDate := targetDate
	var maxDateStr string
	sql := fmt.Sprintf("SELECT COALESCE(MAX(date)::VARCHAR, '') FROM read_parquet('%s') WHERE date < '%s'", bksPath, targetDate)
	database.QueryRow(sql).Scan(&maxDateStr)
	if maxDateStr != "" {
		brokerDate = maxDateStr
	}
	fmt.Printf("  broker date: %s (target: %s)\n", brokerDate, targetDate)

	_, err = features.ComputeBrokerAggregate(database, bksPath, mbPath, brokerDate, targetDate)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Warning: broker (T-1) skipped: %v\n", err)
	}
	_, err = features.ComputeCVD(database, bksPath, brokerDate, targetDate)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Warning: cvd skipped: %v\n", err)
	}

	return nil
}

func upsertLiveMomentum(database *sql.DB, yf1hPath, targetDate string) (int, error) {
	rows, err := features.ComputeMomentum(yf1hPath, targetDate)
	if err != nil {
		return 0, err
	}

	for _, c := range []string{"entry_price", "close_ret_last1h", "close_vs_open_day", "close_range_pct"} {
		if _, err := database.Exec(fmt.Sprintf(`ALTER TABLE features_store ADD COLUMN IF NOT EXISTS "%s" DOUBLE`, c)); err != nil {
			return 0, err
		}
	}

	updated := 0
	for _, r := range rows {
		assignments := []string{}
		args := []interface{}{}
		if r.EntryPrice != nil {
			assignments = append(assignments, `"entry_price" = ?`)
			args = append(args, *r.EntryPrice)
		}
		if r.CloseRetLast1h != nil {
			assignments = append(assignments, `"close_ret_last1h" = ?`)
			args = append(args, *r.CloseRetLast1h)
		}
		if r.CloseVsOpenDay != nil {
			assignments = append(assignments, `"close_vs_open_day" = ?`)
			args = append(args, *r.CloseVsOpenDay)
		}
		if r.CloseRangePct != nil {
			assignments = append(assignments, `"close_range_pct" = ?`)
			args = append(args, *r.CloseRangePct)
		}
		if len(assignments) == 0 {
			continue
		}

		args = append(args, targetDate, r.Ticker)
		res, err := database.Exec(
			fmt.Sprintf(`UPDATE features_store SET %s WHERE date = ? AND ticker = ?`, strings.Join(assignments, ", ")),
			args...,
		)
		if err != nil {
			return updated, err
		}
		if n, _ := res.RowsAffected(); n > 0 {
			updated++
		}
	}
	return updated, nil
}

func upsertLivePreclose14(database *sql.DB, yf1hPath, targetDate string) (int, error) {
	rows, err := features.ComputePreclose14(yf1hPath, targetDate)
	if err != nil {
		return 0, err
	}

	allCols := make(map[string]struct{})
	for _, r := range rows {
		for c := range r.Cols {
			allCols[c] = struct{}{}
		}
	}
	for c := range allCols {
		if _, err := database.Exec(fmt.Sprintf(`ALTER TABLE features_store ADD COLUMN IF NOT EXISTS "%s" DOUBLE`, c)); err != nil {
			return 0, err
		}
	}

	updated := 0
	for _, r := range rows {
		if len(r.Cols) == 0 {
			continue
		}

		colNames := make([]string, 0, len(r.Cols))
		for c := range r.Cols {
			colNames = append(colNames, c)
		}
		sort.Strings(colNames)

		assignments := make([]string, 0, len(colNames))
		args := make([]interface{}, 0, len(colNames)+2)
		for _, c := range colNames {
			assignments = append(assignments, fmt.Sprintf(`"%s" = ?`, c))
			args = append(args, r.Cols[c])
		}

		args = append(args, targetDate, r.Ticker)
		res, err := database.Exec(
			fmt.Sprintf(`UPDATE features_store SET %s WHERE date = ? AND ticker = ?`, strings.Join(assignments, ", ")),
			args...,
		)
		if err != nil {
			return updated, err
		}
		if n, _ := res.RowsAffected(); n > 0 {
			updated++
		}
	}
	return updated, nil
}

// ── download ───────────────────────────────────────────────────────────

type yfBar struct {
	Datetime time.Time `parquet:"datetime"`
	Ticker   string    `parquet:"ticker"`
}

func cmdDownload(repoRoot string, args []string) error {
	f := flag.NewFlagSet("download", flag.ExitOnError)
	limit := f.Int("limit", 0, "Max tickers to download (0=all)")
	date := f.String("date", "today", "Target date")
	f.Parse(args)

	cfg, err := loadConfig(repoRoot)
	if err != nil {
		return err
	}

	targetDate := *date
	if targetDate == "today" {
		targetDate = time.Now().Format("2006-01-02")
	}

	yf1hPath := cfg.L0File("yfinance_1h.parquet")

	// Read existing 1h parquet to get ticker list
	bars, err := parquet.ReadFile[yfBar](yf1hPath)
	if err != nil {
		return fmt.Errorf("read %s: %w", yf1hPath, err)
	}

	tickerSet := make(map[string]bool)
	for _, b := range bars {
		tickerSet[strings.TrimSuffix(b.Ticker, ".JK")] = true
	}

	tickers := make([]string, 0, len(tickerSet))
	for t := range tickerSet {
		tickers = append(tickers, t)
	}
	sort.Strings(tickers)

	if *limit > 0 && *limit < len(tickers) {
		tickers = tickers[:*limit]
	}

	fmt.Printf("Downloading 1h bars for %d tickers (date: %s)...\n", len(tickers), targetDate)
	n, err := fetcher.Download1h(tickers, yf1hPath, targetDate)
	if err != nil {
		return err
	}
	fmt.Printf("Done: %d new bars written\n", n)
	return nil
}

// ── predict ──────────────────────────────────────────────────────────────

func cmdPredict(repoRoot string, args []string) error {
	f := flag.NewFlagSet("predict", flag.ExitOnError)
	variant := f.String("variant", "v15", "Model variant")
	logPicks := f.Bool("log", false, "Log to picks_log")
	date := f.String("date", "", "Target date")
	dbPath := f.String("db", "", "Path to DuckDB (default: config)")
	dumpTicker := f.String("dump-ticker", "", "Dump feature vector for this ticker")
	f.Parse(args)

	cfg, err := loadConfig(repoRoot)
	if err != nil {
		return err
	}

	duckPath := cfg.DuckDBPath
	if *dbPath != "" {
		duckPath = *dbPath
	}

	// Load model
	modelPath := fmt.Sprintf("%s/bsjp_%s/model_lightgbm_opening_tp3.txt", cfg.ModelDir, *variant)
	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		modelPath = fmt.Sprintf("%s/%s/model_lightgbm_opening_tp3.txt", cfg.ModelDir, *variant)
	}
	lgb, err := model.LoadLightGBM(modelPath)
	if err != nil {
		return fmt.Errorf("load model: %w", err)
	}

	// Determine ARA-state policy based on variant
	araPolicy := "none"
	if strings.Contains(*variant, "v19d") || strings.Contains(*variant, "v20") {
		araPolicy = "v20_clean"
	}

	database, err := db.Open(duckPath)
	if err != nil {
		return err
	}
	defer database.Close()

	targetDate := *date
	if targetDate == "" {
		latest, _ := db.GetLatestDate(database)
		if latest == "" {
			return fmt.Errorf("no features in store. run bootstrap or fetch first")
		}
		targetDate = latest
	}

	// Get column names from features_store
	colRows, err := database.Query("SELECT column_name FROM information_schema.columns WHERE table_name='features_store' ORDER BY ordinal_position")
	if err != nil {
		return err
	}
	var allCols []string
	for colRows.Next() {
		var c string
		colRows.Scan(&c)
		allCols = append(allCols, c)
	}
	colRows.Close()

	// Build SELECT with all feature columns
	selectCols := []string{"ticker", "entry_price"}
	for _, c := range allCols {
		if c == "date" || c == "ticker" || c == "entry_price" || c == "inserted_at" {
			continue
		}
		selectCols = append(selectCols, fmt.Sprintf(`"%s"`, c))
	}

	query := fmt.Sprintf("SELECT %s FROM features_store WHERE date = ? AND entry_price > 0", strings.Join(selectCols, ", "))
	rows, err := database.Query(query, targetDate)
	if err != nil {
		return err
	}
	defer rows.Close()

	cols, _ := rows.Columns()
	type pick struct {
		ticker     string
		entryPrice float64
		proba      float64
		features   map[string]float64
	}
	var picks []pick

	for rows.Next() {
		vals := make([]interface{}, len(cols))
		ptrs := make([]interface{}, len(cols))
		for i := range vals {
			ptrs[i] = &vals[i]
		}
		if err := rows.Scan(ptrs...); err != nil {
			return err
		}

		ticker := ""
		var entryPrice float64
		feat := make(map[string]float64)
		for i, c := range cols {
			if c == "ticker" {
				if v, ok := vals[i].(string); ok {
					ticker = v
				}
			} else if c == "entry_price" {
				if v, ok := vals[i].(float64); ok {
					entryPrice = v
				}
			} else if v, ok := toFloat64(vals[i]); ok {
				feat[c] = v
			}
		}

		proba := lgb.Predict(feat)
		picks = append(picks, pick{ticker, entryPrice, proba, feat})

		if *dumpTicker != "" && ticker == *dumpTicker {
			b, _ := json.MarshalIndent(feat, "", "  ")
			fmt.Printf("--- FEATURE VECTOR DUMP: %s ---\n", ticker)
			os.Stdout.Write(b)
			fmt.Println("\n--- END DUMP ---")
		}

		if len(picks) <= 5 {
			fmt.Printf("  debug ticker=%s features=%d entry=%.0f proba=%.4f\n", ticker, len(feat), entryPrice, proba)
		}
	}

	if len(picks) == 0 {
		return fmt.Errorf("no executable features for date %s. run fetch first and ensure entry_price is populated", targetDate)
	}

	sort.Slice(picks, func(i, j int) bool {
		return picks[i].proba > picks[j].proba
	})

	for _, r := range picks {
		if r.ticker == "HRTA" || r.ticker == "GGRM" {
			fmt.Printf("  debug ticker=%s proba=%.4f cost=%.4f ara=%.4f\n", r.ticker, r.proba, r.features["pre14_market_cost_est"], r.features["pre14_ara_touched"])
		}
	}

	// Apply base filters for v23 (price >= 500, cost <= 0.03)
	if strings.Contains(*variant, "v23") {
		filtered := picks[:0]
		for _, p := range picks {
			if cost, ok := p.features["pre14_market_cost_est"]; ok {
				if cost > 0.03 {
					continue
				}
			}
			if p.entryPrice < 500 {
				continue
			}
			filtered = append(filtered, p)
		}
		picks = filtered
	}

	// Apply ARA-state policy filter (v20)
	beforeFilter := len(picks)
	if araPolicy != "none" {
		filtered := picks[:0]
		for _, p := range picks {
			touched := p.features["pre14_ara_touched"]
			wickCnt := p.features["pre14_ara_release_wick_count"]
			araDist := p.features["pre14_ara_distance_pct"]
			keep := false
			// single release wick: ARA touched + exactly 1 release
			if touched >= 0.5 && wickCnt >= 0.5 && wickCnt <= 1.5 {
				keep = true
			}
			// near ARA not touched: within 0-3%
			if touched < 0.5 && araDist > 0 && araDist <= 0.03 {
				keep = true
			}
			if keep {
				filtered = append(filtered, p)
			}
		}
		if len(filtered) > 0 {
			picks = filtered
		}
		fmt.Printf("  ARA policy: %d/%d picks kept (policy=%s)\n", len(picks), beforeFilter, araPolicy)
	}

	topN := 3
	if strings.Contains(*variant, "v23") {
		topN = 2
	}
	if topN > len(picks) {
		topN = len(picks)
	}

	fmt.Printf("Top-%d picks for %s (model: %s):\n", topN, targetDate, *variant)
	for j := 0; j < min(20, len(picks)); j++ {
		fmt.Printf("  #%d %s  proba=%.4f  entry=%.0f  features=%d\n", j+1, picks[j].ticker, picks[j].proba, picks[j].entryPrice, len(picks[j].features))
	}

	if *logPicks {
		var dbPicks []db.PickRow
		for i := 0; i < topN; i++ {
			ep := picks[i].entryPrice
			proba := picks[i].proba
			dbPicks = append(dbPicks, db.PickRow{
				Date:       targetDate,
				Variant:    *variant,
				Rank:       i + 1,
				Ticker:     picks[i].ticker,
				PredProba:  &proba,
				EntryPrice: &ep,
			})
		}
		if err := db.LogPicks(database, dbPicks); err != nil {
			return err
		}
		fmt.Printf("Logged to picks_log (variant: %s)\n", *variant)
	}
	return nil
}

func toFloat64(v interface{}) (float64, bool) {
	switch x := v.(type) {
	case float64:
		return x, true
	case int64:
		return float64(x), true
	case float32:
		return float64(x), true
	case nil:
		return 0, false
	}
	return 0, false
}

// ── check ────────────────────────────────────────────────────────────────

func cmdCheck(repoRoot string, args []string) error {
	f := flag.NewFlagSet("check", flag.ExitOnError)
	verbose := f.Bool("verbose", false, "Detailed output")
	telegram := f.Bool("telegram", false, "Send notification to Telegram")
	dbPath := f.String("db", "", "Path to DuckDB")
	f.Parse(args)
	return cmdCheckWithDB(repoRoot, *dbPath, *verbose, *telegram)
}

func cmdCheckWithDB(repoRoot string, dbPath string, verbose bool, tgFlag bool) error {
	cfg, err := loadConfig(repoRoot)
	if err != nil {
		return err
	}

	duckPath := cfg.DuckDBPath
	if dbPath != "" {
		duckPath = dbPath
	}

	loc, _ := time.LoadLocation("Asia/Jakarta")
	nowWIB := time.Now().In(loc)
	today := nowWIB.Format("2006-01-02")
	isWeekend := nowWIB.Weekday() == time.Saturday || nowWIB.Weekday() == time.Sunday

	checkDB, _ := db.Open(duckPath)
	tMinus1 := prevTradingDay(nowWIB).Format("2006-01-02")

	// 09:30 WIB rule for DuckDB today's data
	needDBDate := tMinus1
	if nowWIB.Hour() > 9 || (nowWIB.Hour() == 9 && nowWIB.Minute() >= 30) {
		needDBDate = today
	}

	type item struct {
		Name   string
		Status string
		Detail string
	}
	var items []item
	allOK := true

	// ── 0. Market Context ──
	contextLine := fmt.Sprintf("D-Day: %s | T-1: %s", today, tMinus1)
	if isWeekend {
		contextLine += " (Weekend 🛌)"
	}

	// ── 1. L0 Parquet Readiness (T-1 Check) ──
	l0Files := []struct {
		label, path, dateExpr, needDate, entityLabel, entityCol string
	}{
		{"L0_broksum", cfg.L0File("broksum_bybroker.parquet"), "date", tMinus1, "tickers", "stock_code"},
		{"L0_yf_daily", cfg.L0File("yfinance_daily.parquet"), "date", tMinus1, "tickers", "ticker"},
		{"L0_yf_1h", cfg.L0File("yfinance_1h.parquet"), "datetime", today, "tickers", "ticker"},
		{"L0_global", cfg.L0File("global_indices.parquet"), "date", tMinus1, "symbols", "symbol"},
	}

	for _, l0 := range l0Files {
		if _, err := os.Stat(l0.path); os.IsNotExist(err) {
			items = append(items, item{l0.label, "❌", "missing"})
			allOK = false
			continue
		}
		dataDate, rows, entities := "", int64(0), int64(0)
		maxHour := -1
		if checkDB != nil {
			if l0.label == "L0_yf_1h" {
				dataDate, rows, entities, maxHour = yf1hLatestQuality(l0.path, loc)
			} else {
				dataDate, rows, entities = parquetLatestQuality(checkDB, l0.path, l0.dateExpr, l0.entityCol)
			}
		}
		quality := freshnessQuality(dataDate, l0.needDate)
		if l0.label == "L0_yf_1h" && quality == "OK" && maxHour >= 0 && maxHour < 15 {
			quality = "NO_15_BAR"
		}
		detail := fmt.Sprintf(
			"latest=%s need=%s rows=%d %s=%d quality=%s",
			emptyDash(dataDate), l0.needDate, rows, l0.entityLabel, entities, quality,
		)
		if l0.label == "L0_yf_1h" && maxHour >= 0 {
			detail = fmt.Sprintf("%s max_hour=%d", detail, maxHour)
		}
		if l0.label == "L0_broksum" && checkDB != nil && dataDate != "" {
			brokers := parquetDistinctOnDate(checkDB, l0.path, l0.dateExpr, "broker", dataDate)
			detail = fmt.Sprintf("%s brokers=%d", detail, brokers)
		}
		if dataDate < l0.needDate {
			items = append(items, item{l0.label, "❌", detail})
			allOK = false
		} else if l0.label == "L0_yf_1h" && quality == "NO_15_BAR" {
			items = append(items, item{l0.label, "❌", detail})
			allOK = false
		} else {
			items = append(items, item{l0.label, "✅", detail})
		}
	}

	// ── 2. DuckDB Features & Integrity (The "Ready-Ready" Check) ──
	if checkDB != nil {
		var count, withBroker, withPrice int64
		var maxDate *time.Time
		checkDB.QueryRow("SELECT COUNT(*), MAX(date) FROM features_store").Scan(&count, &maxDate)

		maxDateStr := ""
		if maxDate != nil {
			maxDateStr = maxDate.Format("2006-01-02")
		}

		// Deep Integrity: Do we have non-null executable features for the required DB date?
		checkDB.QueryRow("SELECT COUNT(*) FROM features_store WHERE date = ? AND flow_total_net_buy_sum IS NOT NULL", needDBDate).Scan(&withBroker)
		checkDB.QueryRow("SELECT COUNT(*) FROM features_store WHERE date = ? AND entry_price > 0", needDBDate).Scan(&withPrice)

		dbQuality := freshnessQuality(maxDateStr, needDBDate)
		if dbQuality == "OK" && nowWIB.Hour() >= 9 && withPrice == 0 {
			dbQuality = "NO_ENTRY"
		} else if dbQuality == "OK" && nowWIB.Hour() >= 9 && withBroker == 0 {
			dbQuality = "NO_BROKER"
		}
		dbDetail := fmt.Sprintf(
			"latest=%s need=%s rows_total=%d entry_rows=%d broker_rows=%d quality=%s",
			emptyDash(maxDateStr), needDBDate, count, withPrice, withBroker, dbQuality,
		)
		if maxDateStr < needDBDate {
			items = append(items, item{"DB_Sync", "❌", dbDetail})
			allOK = false
		} else if nowWIB.Hour() >= 9 && withPrice == 0 {
			items = append(items, item{"DB_Price", "❌", dbDetail})
			allOK = false
		} else if nowWIB.Hour() >= 9 && withBroker == 0 {
			items = append(items, item{"DB_Brok", "⚠️", dbDetail})
		} else {
			items = append(items, item{"DB_State", "✅", dbDetail})
		}
		checkDB.Close()
	}

	// ── 3. Model Check ──
	mpath := fmt.Sprintf("%s/v25_clean_t1quick_nl31_md100_l2.0_market_k3_w25/model_lightgbm_opening_tp3.txt", cfg.ModelDir)
	if _, err := os.Stat(mpath); err == nil {
		trees, feats := modelCounts(mpath)
		items = append(items, item{"Model", "✅", fmt.Sprintf("v25 loaded trees=%d features=%d quality=OK", trees, feats)})
	} else {
		items = append(items, item{"Model", "❌", "missing"})
		allOK = false
	}

	// ── Output Building ──
	fmt.Printf("\n--- %s ---\n", contextLine)
	sb := &strings.Builder{}
	sb.WriteString(fmt.Sprintf("🔍 *BSJP Heartbeat %s*\n`%s`\n\n", nowWIB.Format("15:04 WIB"), contextLine))

	for _, it := range items {
		line := fmt.Sprintf("%s %-12s %s\n", it.Status, it.Name, it.Detail)
		fmt.Print(line)
		sb.WriteString(fmt.Sprintf("%s %s `%s`\\n", it.Status, it.Name, it.Detail))
	}

	if allOK && !isWeekend {
		sb.WriteString("\n✅ *SYSTEM READY-READY 🚀*")
	} else if isWeekend {
		sb.WriteString("\n🛌 *MARKET CLOSED (Enjoy your weekend)*")
	} else {
		sb.WriteString("\n🚨 *ISSUES DETECTED - CHECK LOGS*")
	}

	if tgFlag {
		token := os.Getenv("BSJP_TELEGRAM_TOKEN")
		chatID := os.Getenv("BSJP_TELEGRAM_CHAT_ID")
		if token != "" && chatID != "" {
			notify.Send(token, chatID, sb.String())
		}
		// Discord...
		dcToken := os.Getenv("BSJP_DISCORD_TOKEN")
		dcChannel := os.Getenv("BSJP_DISCORD_CHANNEL")
		if dcToken != "" && dcChannel != "" {
			notify.SendDiscordPreflight(dcToken, dcChannel, "BSJP Audit", nil, allOK) // Simplified
		}
	}

	if !allOK && !isWeekend {
		return fmt.Errorf("preflight integrity check failed")
	}
	return nil
}

// humanSize formats a file size in human-readable form.
func humanSize(n int64) string {
	units := []string{"B", "KB", "MB", "GB"}
	v := float64(n)
	i := 0
	for v >= 1024 && i < len(units)-1 {
		v /= 1024
		i++
	}
	return fmt.Sprintf("%.1f%s", v, units[i])
}

func duckQuote(s string) string {
	return strings.ReplaceAll(s, "'", "''")
}

func emptyDash(s string) string {
	if s == "" {
		return "-"
	}
	return s
}

func freshnessQuality(latest, need string) string {
	if latest == "" {
		return "NO_DATA"
	}
	if latest < need {
		return "STALE"
	}
	if latest > need {
		return "AHEAD"
	}
	return "OK"
}

func parquetLatestQuality(database *sql.DB, path, dateExpr, entityCol string) (latest string, rows, entities int64) {
	entityExpr := "''"
	if entityCol != "" {
		entityExpr = entityCol
	}
	query := fmt.Sprintf(`
		WITH src AS (
			SELECT CAST(%s AS DATE) AS d, CAST(%s AS VARCHAR) AS entity
			FROM read_parquet('%s')
		),
		latest AS (
			SELECT MAX(d) AS max_d FROM src
		)
		SELECT
			COALESCE(max_d::VARCHAR, ''),
			COUNT(*) FILTER (WHERE d = max_d),
			COUNT(DISTINCT entity) FILTER (WHERE d = max_d)
		FROM src, latest
		GROUP BY max_d
	`, dateExpr, entityExpr, duckQuote(path))
	_ = database.QueryRow(query).Scan(&latest, &rows, &entities)
	return latest, rows, entities
}

func parquetDistinctOnDate(database *sql.DB, path, dateExpr, entityCol, date string) int64 {
	var n int64
	query := fmt.Sprintf(`
		SELECT COUNT(DISTINCT %s)
		FROM read_parquet('%s')
		WHERE CAST(%s AS DATE) = DATE '%s'
	`, entityCol, duckQuote(path), dateExpr, date)
	_ = database.QueryRow(query).Scan(&n)
	return n
}

func yf1hLatestQuality(path string, loc *time.Location) (latest string, rows, tickers int64, maxHour int) {
	maxHour = -1
	bars, err := parquet.ReadFile[yfBar](path)
	if err != nil || len(bars) == 0 {
		return "", 0, 0, maxHour
	}
	for _, b := range bars {
		ds := b.Datetime.In(loc).Format("2006-01-02")
		if ds > latest {
			latest = ds
		}
	}
	seen := make(map[string]struct{})
	for _, b := range bars {
		if b.Datetime.In(loc).Format("2006-01-02") != latest {
			continue
		}
		rows++
		h := b.Datetime.In(loc).Hour()
		if h > maxHour {
			maxHour = h
		}
		seen[strings.TrimSuffix(b.Ticker, ".JK")] = struct{}{}
	}
	return latest, rows, int64(len(seen)), maxHour
}

// prevTradingDay returns the most recent weekday before t (skips t itself).
func prevTradingDay(t time.Time) time.Time {
	d := t.AddDate(0, 0, -1)
	for {
		wd := d.Weekday()
		if wd >= time.Monday && wd <= time.Friday {
			return d
		}
		d = d.AddDate(0, 0, -1)
	}
}

// modelCounts reads tree count and feature count from a LightGBM model file header.
func modelCounts(path string) (trees, features int) {
	f, err := os.Open(path)
	if err != nil {
		return 0, 0
	}
	defer f.Close()

	// Read first 5 lines to find feature_names and tree_sizes
	scanner := bufio.NewScanner(f)
	scanner.Buffer(make([]byte, 1<<20), 1<<24)
	lineCount := 0
	for scanner.Scan() && lineCount < 15 {
		line := scanner.Text()
		lineCount++
		if strings.HasPrefix(line, "feature_names=") {
			s := line[len("feature_names="):]
			features = len(strings.Fields(s))
		}
		if strings.HasPrefix(line, "tree_sizes=") {
			s := line[len("tree_sizes="):]
			trees = len(strings.Fields(s))
		}
	}
	return trees, features
}

func f64Ptr(v float64) *float64 {
	if math.IsNaN(v) || math.IsInf(v, 0) {
		return nil
	}
	return &v
}

func indexBroker(rows []features.BrokerRow) map[string]*features.BrokerRow {
	m := make(map[string]*features.BrokerRow)
	for i := range rows {
		m[rows[i].Ticker] = &rows[i]
	}
	return m
}

func indexOvernight(rows []features.OvernightRow) map[string]*features.OvernightRow {
	m := make(map[string]*features.OvernightRow)
	for i := range rows {
		m[rows[i].Ticker] = &rows[i]
	}
	return m
}

func setBrokerCols(dr *db.FeatureRow, b *features.BrokerRow) {
	dr.Set("flow_total_net_buy_sum", b.FlowTotalNetBuySum)
	dr.Set("flow_total_net_buy_mean", b.FlowTotalNetBuyMean)
	dr.Set("flow_gross_turnover_sum", b.FlowGrossTurnoverSum)
	dr.Set("flow_gross_turnover_mean", b.FlowGrossTurnoverMean)
	dr.Set("flow_buy_freq_sum", b.FlowBuyFreqSum)
	dr.Set("flow_buy_freq_mean", b.FlowBuyFreqMean)
	dr.Set("flow_sell_freq_sum", b.FlowSellFreqSum)
	dr.Set("flow_sell_freq_mean", b.FlowSellFreqMean)
	dr.Set("flow_abs_net_buy_sum", b.FlowAbsNetBuySum)
	dr.Set("flow_abs_net_buy_mean", b.FlowAbsNetBuyMean)
	dr.Set("flow_net_flow_ratio_sum", b.FlowNetFlowRatioSum)
	dr.Set("flow_net_flow_ratio_mean", b.FlowNetFlowRatioMean)
	dr.Set("flow_total_trades_sum", b.FlowTotalTradesSum)
	dr.Set("flow_total_trades_mean", b.FlowTotalTradesMean)
	dr.Set("flow_net_buy_per_trade_sum", b.FlowNetBuyPerTradeSum)
	dr.Set("flow_net_buy_per_trade_mean", b.FlowNetBuyPerTradeMean)
	dr.Set("flow_churn_ratio_sum", b.FlowChurnRatioSum)
	dr.Set("flow_churn_ratio_mean", b.FlowChurnRatioMean)
	dr.Set("ctx_broker_ticker_specificity_sum", b.CtxTickerSpecificitySum)
	dr.Set("ctx_broker_ticker_specificity_mean", b.CtxTickerSpecificityMean)
	dr.Set("ctx_broker_market_share_sum", b.CtxMarketShareSum)
	dr.Set("ctx_broker_market_share_mean", b.CtxMarketShareMean)
	dr.Set("ctx_ticker_market_share_sum", b.CtxTickerMarketShareSum)
	dr.Set("ctx_ticker_market_share_mean", b.CtxTickerMarketShareMean)
	dr.Set("ctx_broker_net_buy_rank_sum", b.CtxNetBuyRankSum)
	dr.Set("ctx_broker_net_buy_rank_mean", b.CtxNetBuyRankMean)
}

func setOvernightCols(dr *db.FeatureRow, o *features.OvernightRow) {
	for col, val := range o.Cols {
		v := val
		dr.Cols[col] = &v
	}
}

func setGlobalCols(dr *db.FeatureRow, g *features.GlobalRow) {
	dr.Set("ihsg_prev_close", g.IhsgPrevClose)
	dr.Set("ihsg_prev_return", g.IhsgPrevReturn)
	dr.Set("ihsg_ma_5", g.IhsgMA5)
	dr.Set("ihsg_ma_20", g.IhsgMA20)
	dr.Set("ihsg_ma_100", g.IhsgMA100)
	dr.Set("ihsg_ma_200", g.IhsgMA200)
	dr.Set("ihsg_close_ma5_ratio", g.IhsgCloseMA5Ratio)
	dr.Set("ihsg_close_ma20_ratio", g.IhsgCloseMA20Ratio)
	dr.Set("nasdaq_prev_return", g.NasdaqPrevReturn)
	dr.Set("nikkei_prev_return", g.NikkeiPrevReturn)
	dr.Set("vix_prev_close", g.VixPrevClose)
	dr.Set("vix_5d_avg", g.Vix5dAvg)
	dr.Set("usdidr_prev_close", g.UsdidrPrevClose)
	dr.Set("usdidr_prev_return", g.UsdidrPrevReturn)
	dr.Set("usdidr_5d_return", g.Usdidr5dReturn)
}

func indexYf(rows []features.YfRow) map[string]*features.YfRow {
	m := make(map[string]*features.YfRow)
	for i := range rows {
		m[rows[i].Ticker] = &rows[i]
	}
	return m
}

func setYfCols(dr *db.FeatureRow, y *features.YfRow) {
	for col, val := range y.Cols {
		v := val
		dr.Cols[col] = &v
	}
}

func indexPreclose14(rows []features.Preclose14Row) map[string]*features.Preclose14Row {
	m := make(map[string]*features.Preclose14Row)
	for i := range rows {
		m[rows[i].Ticker] = &rows[i]
	}
	return m
}

func setPreclose14Cols(dr *db.FeatureRow, p *features.Preclose14Row) {
	for col, val := range p.Cols {
		v := val
		dr.Cols[col] = &v
	}
}

func indexXl(rows []features.XlRow) map[string]*features.XlRow {
	m := make(map[string]*features.XlRow)
	for i := range rows {
		m[rows[i].Ticker] = &rows[i]
	}
	return m
}

func setXlCols(dr *db.FeatureRow, x *features.XlRow) {
	for col, val := range x.Cols {
		v := val
		dr.Cols[col] = &v
	}
}

func indexAraHistory(rows []features.AraHistoryRow) map[string]*features.AraHistoryRow {
	m := make(map[string]*features.AraHistoryRow)
	for i := range rows {
		m[rows[i].Ticker] = &rows[i]
	}
	return m
}

func setAraHistoryCols(dr *db.FeatureRow, a *features.AraHistoryRow) {
	for col, val := range a.Cols {
		v := val
		dr.Cols[col] = &v
	}
}
