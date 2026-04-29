package features

import (
	"fmt"
	"math"
	"sort"
	"strings"
	"time"

	"github.com/parquet-go/parquet-go"
)

// ── Parquet schemas ──────────────────────────────────────────────────────────

type broksumRow struct {
	Broker    string  `parquet:"broker"`
	StockCode string  `parquet:"stock_code"`
	Date      string  `parquet:"date"`
	NetVal    float64 `parquet:"net_val"`
	TotalVal  float64 `parquet:"total_val"`
	BuyFreq   int64   `parquet:"buy_freq"`
	SellFreq  int64   `parquet:"sell_freq"`
}

// BrokerRow holds broker features for the L2 table (exact column names).
type BrokerRow struct {
	Date   string
	Ticker string

	FlowTotalNetBuySum  float64
	FlowTotalNetBuyMean float64
	FlowGrossTurnoverSum  float64
	FlowGrossTurnoverMean float64
	FlowBuyFreqSum      float64
	FlowBuyFreqMean     float64
	FlowSellFreqSum     float64
	FlowSellFreqMean    float64
	FlowAbsNetBuySum    float64
	FlowAbsNetBuyMean   float64
	FlowNetFlowRatioSum float64
	FlowNetFlowRatioMean float64
	FlowTotalTradesSum  float64
	FlowTotalTradesMean float64
	FlowNetBuyPerTradeSum  float64
	FlowNetBuyPerTradeMean float64
	FlowChurnRatioSum   float64
	FlowChurnRatioMean  float64

	CtxTickerSpecificitySum  float64
	CtxTickerSpecificityMean float64
	CtxMarketShareSum        float64
	CtxMarketShareMean       float64
	CtxTickerMarketShareSum  float64
	CtxTickerMarketShareMean float64
	CtxNetBuyRankSum         float64
	CtxNetBuyRankMean        float64
}

// ComputeBrokerFeatures computes broker flow + context features at L2 grain.
// Applies shift(1): features at date T use data ≤ T-1.
func ComputeBrokerFeatures(broksumPath, masterBrokerPath, targetDate string) ([]BrokerRow, error) {
	rows, err := parquet.ReadFile[broksumRow](broksumPath)
	if err != nil {
		return nil, fmt.Errorf("read broksum: %w", err)
	}

	t, _ := time.Parse("2006-01-02", targetDate)
	tMinus1 := t.AddDate(0, 0, -1).Format("2006-01-02")
	tMinus60 := t.AddDate(0, 0, -60).Format("2006-01-02")

	// Per (date, broker, ticker) stats
	type key struct{ broker, ticker, date string }
	type stat struct {
		netVal, totalVal float64
		buyFreq, sellFreq int64
	}
	stats := make(map[key]*stat)

	for _, r := range rows {
		if r.Date < tMinus60 || r.Date > tMinus1 {
			continue
		}
		k := key{r.Broker, r.StockCode, r.Date}
		s := stats[k]
		if s == nil {
			s = &stat{}
			stats[k] = s
		}
		s.netVal += r.NetVal
		s.totalVal += r.TotalVal
		s.buyFreq += r.BuyFreq
		s.sellFreq += r.SellFreq
	}

	// Context aggregations
	brokerDayFlow := make(map[string]float64) // "date|broker" → sum(|net_val|)
	tickerDayFlow := make(map[string]float64) // "date|ticker" → sum(|net_val|)
	marketDayFlow := make(map[string]float64) // date → sum(|net_val|)

	for k, s := range stats {
		absNet := math.Abs(s.netVal)
		brokerDayFlow[k.date+"|"+k.broker] += absNet
		tickerDayFlow[k.date+"|"+k.ticker] += absNet
		marketDayFlow[k.date] += absNet
	}

	// Per (date, ticker) aggregation with all per-broker stats accumulated in one pass
	type l2Key struct{ date, ticker string }
	type l2Acc struct {
		netValSum, totalValSum       float64
		buyFreqSum, sellFreqSum      int64
		absNetSum                    float64
		netFlowRatioSum, churnSum    float64
		totalTradesSum, nbptSum      float64
		brokerCount                  int
		specificities, marketShares  []float64
	}
	l2 := make(map[l2Key]*l2Acc)

	for k, s := range stats {
		lk := l2Key{k.date, k.ticker}
		a := l2[lk]
		if a == nil {
			a = &l2Acc{}
			l2[lk] = a
		}
		a.netValSum += s.netVal
		a.totalValSum += s.totalVal
		a.buyFreqSum += s.buyFreq
		a.sellFreqSum += s.sellFreq
		a.absNetSum += math.Abs(s.netVal)
		a.brokerCount++

		if s.totalVal != 0 {
			a.netFlowRatioSum += s.netVal / s.totalVal
			abs := math.Abs(s.netVal)
			if abs > 0 {
				a.churnSum += s.totalVal / abs
			}
		}
		trades := float64(s.buyFreq + s.sellFreq)
		a.totalTradesSum += trades
		if trades > 0 {
			a.nbptSum += s.netVal / trades
		}

		absNet := math.Abs(s.netVal)
		if tf, ok := tickerDayFlow[k.date+"|"+k.ticker]; ok && tf > 0 {
			a.specificities = append(a.specificities, absNet/tf)
		}
		if bf, ok := brokerDayFlow[k.date+"|"+k.broker]; ok {
			if mf, ok := marketDayFlow[k.date]; ok && mf > 0 {
				a.marketShares = append(a.marketShares, bf/mf)
			}
		}
	}

	// Precompute net_buy_rank per (date, ticker) — one pass, not O(n²)
	dateTickerNetVals := make(map[string]map[string][]float64)
	for k, s := range stats {
		if dateTickerNetVals[k.date] == nil {
			dateTickerNetVals[k.date] = make(map[string][]float64)
		}
		dateTickerNetVals[k.date][k.ticker] = append(dateTickerNetVals[k.date][k.ticker], s.netVal)
	}
	dateTickerRank := make(map[string]map[string]float64)
	for date, tickers := range dateTickerNetVals {
		dateTickerRank[date] = make(map[string]float64)
		for ticker, vals := range tickers {
			sort.Float64s(vals)
			if len(vals) > 0 {
				dateTickerRank[date][ticker] = float64(len(vals)/2) / float64(len(vals))
			}
		}
	}

	// Build result rows
	var result []BrokerRow
	for lk, a := range l2 {
		n := float64(a.brokerCount)
		if n == 0 {
			continue
		}
		r := BrokerRow{Date: lk.date, Ticker: lk.ticker}

		r.FlowTotalNetBuySum = a.netValSum
		r.FlowTotalNetBuyMean = a.netValSum / n
		r.FlowGrossTurnoverSum = a.totalValSum
		r.FlowGrossTurnoverMean = a.totalValSum / n
		r.FlowBuyFreqSum = float64(a.buyFreqSum)
		r.FlowBuyFreqMean = float64(a.buyFreqSum) / n
		r.FlowSellFreqSum = float64(a.sellFreqSum)
		r.FlowSellFreqMean = float64(a.sellFreqSum) / n
		r.FlowAbsNetBuySum = a.absNetSum
		r.FlowAbsNetBuyMean = a.absNetSum / n
		r.FlowNetFlowRatioSum = a.netFlowRatioSum
		r.FlowNetFlowRatioMean = a.netFlowRatioSum / n
		r.FlowChurnRatioSum = a.churnSum
		r.FlowChurnRatioMean = a.churnSum / n
		r.FlowTotalTradesSum = a.totalTradesSum
		r.FlowTotalTradesMean = a.totalTradesSum / n
		r.FlowNetBuyPerTradeSum = a.nbptSum
		r.FlowNetBuyPerTradeMean = a.nbptSum / n

		r.CtxTickerSpecificitySum = avg(a.specificities)
		r.CtxTickerSpecificityMean = avg(a.specificities)
		r.CtxMarketShareSum = avg(a.marketShares)
		r.CtxMarketShareMean = avg(a.marketShares)

		if tf, ok := tickerDayFlow[lk.date+"|"+lk.ticker]; ok {
			if mf, ok := marketDayFlow[lk.date]; ok && mf > 0 {
				ts := tf / mf
				r.CtxTickerMarketShareSum = ts
				r.CtxTickerMarketShareMean = ts
			}
		}

		if ranks, ok := dateTickerRank[lk.date]; ok {
			if rank, ok := ranks[lk.ticker]; ok {
				r.CtxNetBuyRankSum = rank
				r.CtxNetBuyRankMean = rank
			}
		}

		result = append(result, r)
	}

	return result, nil
}

func avg(vals []float64) float64 {
	if len(vals) == 0 {
		return 0
	}
	s := 0.0
	for _, v := range vals {
		s += v
	}
	return s / float64(len(vals))
}

func split2(s, sep string) []string {
	return strings.SplitN(s, sep, 2)
}
