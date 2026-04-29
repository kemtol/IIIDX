package main

import (
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
	"github.com/mmmachine/bsjp/internal/model"
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
	case "check":
		return cmdCheck(repoRoot)
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
	if _, err := os.Stat(yf1h); os.IsNotExist(err) {
		return fmt.Errorf("yfinance_1h.parquet not found at %s", yf1h)
	}

	fmt.Printf("Computing features for %s...\n", targetDate)

	// Compute momentum features
	momRows, err := features.ComputeMomentum(yf1h, targetDate)
	if err != nil {
		return err
	}

	// Compute broker features from L0 broksum_bybroker.parquet (uses T-1)
	bksPath := cfg.L0File("broksum_bybroker.parquet")
	mbPath := cfg.L0File("master_broker.parquet")
	tMinus1 := time.Now().AddDate(0, 0, -1).Format("2006-01-02")
	if d, err := time.Parse("2006-01-02", targetDate); err == nil {
		tMinus1 = d.AddDate(0, 0, -1).Format("2006-01-02")
	}
	_, err = features.ComputeBrokerAggregate(database, bksPath, mbPath, tMinus1, targetDate)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Warning: broker (T-1): %v\n", err)
	}

	// Compute overnight features
	overnightRows, err := features.ComputeOvernight(yf1h, targetDate)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Warning: overnight skipped: %v\n", err)
		overnightRows = nil
	}

	// Compute global macro features (per-date, broadcast)
	globalRow, err := features.ComputeGlobal(cfg.L0File("global_indices.parquet"), targetDate)
	if err != nil {
		globalRow = nil
	}

	// Compute OHLCV derived features
	yfRows, err := features.ComputeYfDaily(cfg.L0File("yfinance_daily.parquet"), targetDate)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Warning: yf_daily skipped: %v\n", err)
		yfRows = nil
	}

	// Index by ticker
	overnightByTicker := indexOvernight(overnightRows)
	yfByTicker := indexYf(yfRows)

	// Merge into db rows
	dbRows := make([]db.FeatureRow, 0, len(momRows))
	for _, r := range momRows {
		dr := db.NewFeatureRow(r.Ticker)
		dr.SetPtr("entry_price", r.EntryPrice)
		dr.SetPtr("close_ret_last1h", r.CloseRetLast1h)
		dr.SetPtr("close_vs_open_day", r.CloseVsOpenDay)
		dr.SetPtr("close_range_pct", r.CloseRangePct)

		// Overnight features
		if o, ok := overnightByTicker[r.Ticker]; ok {
			setOvernightCols(&dr, o)
		}

		// OHLCV derived features
		if y, ok := yfByTicker[r.Ticker]; ok {
			setYfCols(&dr, y)
		}

		// Global features (broadcast to all tickers)
		if globalRow != nil {
			setGlobalCols(&dr, globalRow)
		}

		dbRows = append(dbRows, dr)
	}

	n, err := db.UpsertDate(database, targetDate, dbRows)
	if err != nil {
		return err
	}
	fmt.Printf("Done: %d tickers upserted for %s\n", n, targetDate)
	return nil
}

// ── predict ──────────────────────────────────────────────────────────────

func cmdPredict(repoRoot string, args []string) error {
	f := flag.NewFlagSet("predict", flag.ExitOnError)
	variant := f.String("variant", "v15", "Model variant")
	logPicks := f.Bool("log", false, "Log to picks_log")
	date := f.String("date", "", "Target date")
	dbPath := f.String("db", "", "Path to DuckDB (default: config)")
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
	lgb, err := model.LoadLightGBM(modelPath)
	if err != nil {
		return fmt.Errorf("load model: %w", err)
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

	query := fmt.Sprintf("SELECT %s FROM features_store WHERE date = ?", strings.Join(selectCols, ", "))
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
		if len(picks) <= 5 {
			fmt.Printf("  debug ticker=%s features=%d entry=%.0f proba=%.4f\n", ticker, len(feat), entryPrice, proba)
		}
	}

	if len(picks) == 0 {
		return fmt.Errorf("no features for date %s. run fetch first", targetDate)
	}

	sort.Slice(picks, func(i, j int) bool {
		return picks[i].proba > picks[j].proba
	})

	topN := 3
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

func cmdCheck(repoRoot string) error {
	return cmdCheckWithDB(repoRoot, "")
}

func cmdCheckWithDB(repoRoot string, dbPath string) error {
	cfg, err := loadConfig(repoRoot)
	if err != nil {
		return err
	}

	duckPath := cfg.DuckDBPath
	if dbPath != "" {
		duckPath = dbPath
	}

	fmt.Printf("Repo root:    %s\n", cfg.RepoRoot)
	fmt.Printf("L0 dir:       %s\n", cfg.L0Dir)
	fmt.Printf("L2 parquet:   %s\n", cfg.L2Parquet)
	fmt.Printf("DuckDB:       %s\n", duckPath)
	fmt.Printf("Model dir:    %s\n", cfg.ModelDir)

	if _, err := os.Stat(duckPath); err == nil {
		database, err := db.Open(duckPath)
		if err != nil {
			return err
		}
		defer database.Close()
		return db.Check(database)
	}
	fmt.Println("DuckDB not found. Run bootstrap first.")
	return nil
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
	dr.Set("overnight_ret_ma5", o.OvernightMA5)
	dr.Set("overnight_ret_ma20", o.OvernightMA20)
	dr.Set("overnight_ret_ma60", o.OvernightMA60)
	dr.Set("overnight_positive_rate5", o.OvernightPositiveRate5)
	dr.Set("overnight_positive_rate20", o.OvernightPositiveRate20)
	dr.Set("overnight_positive_rate60", o.OvernightPositiveRate60)
	dr.Set("overnight_worst_5d", o.OvernightWorst5)
	dr.Set("overnight_worst_20d", o.OvernightWorst20)
	dr.Set("overnight_worst_60d", o.OvernightWorst60)
	dr.Set("overnight_p10_5d", o.OvernightP10_5)
	dr.Set("overnight_p10_20d", o.OvernightP10_20)
	dr.Set("overnight_p10_60d", o.OvernightP10_60)
	dr.Set("gapdown_freq_5d", o.GapdownFreq5)
	dr.Set("gapdown_freq_20d", o.GapdownFreq20)
	dr.Set("gapdown_freq_60d", o.GapdownFreq60)
	dr.Set("gapdown_severe_freq_5d", o.GapdownSevereFreq5)
	dr.Set("gapdown_severe_freq_20d", o.GapdownSevereFreq20)
	dr.Set("gapdown_severe_freq_60d", o.GapdownSevereFreq60)
	dr.Set("gap_up2_freq_5d", o.GapUp2Freq5)
	dr.Set("gap_up2_freq_20d", o.GapUp2Freq20)
	dr.Set("gap_up2_freq_60d", o.GapUp2Freq60)
	dr.Set("gap_down2_freq_5d", o.GapDown2Freq5)
	dr.Set("gap_down2_freq_20d", o.GapDown2Freq20)
	dr.Set("gap_down2_freq_60d", o.GapDown2Freq60)
	dr.Set("gap_up2_down2_edge_5d", o.GapUp2Down2Edge5)
	dr.Set("gap_up2_down2_edge_20d", o.GapUp2Down2Edge20)
	dr.Set("gap_up2_down2_edge_60d", o.GapUp2Down2Edge60)
	dr.Set("gap_up2_followthrough_freq_5d", o.GapUp2FollowthroughFreq5)
	dr.Set("gap_up2_followthrough_freq_20d", o.GapUp2FollowthroughFreq20)
	dr.Set("gap_up2_followthrough_freq_60d", o.GapUp2FollowthroughFreq60)
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
