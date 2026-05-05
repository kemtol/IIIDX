package features

import (
	"fmt"
	"math"
	"sort"
	"strings"
	"time"

	"github.com/parquet-go/parquet-go"
)

const (
	p14RoundtripCostFrac = 0.004
	p14AraBufferPct      = 0.005
)

var p14PrecloseHours = map[int]bool{9: true, 10: true, 11: true, 13: true, 14: true}
var p14MorningHours = map[int]bool{9: true, 10: true, 11: true}

type p14Bar struct {
	Date   time.Time `parquet:"datetime"`
	Ticker string    `parquet:"ticker"`
	Open   float64   `parquet:"open"`
	High   float64   `parquet:"high"`
	Low    float64   `parquet:"low"`
	Close  float64   `parquet:"close"`
	Volume float64   `parquet:"volume"`
}

type p14DailyAgg struct {
	date    time.Time
	open9   float64
	open14  float64
	close14 float64
	close11 float64
	high    float64
	low     float64
	tpVol   float64
	vol     float64
	vol14h  float64

	orbOpen  float64
	orbHigh  float64
	orbLow   float64
	orbClose float64
	orbHour  int

	amHigh  float64
	amLow   float64
	pmHigh  float64
	pmLow   float64
	openPM  float64
	closePM float64

	amTpVol float64
	amVol   float64

	avgHourly float64
	prevClose float64

	hours   []int
	opens   []float64
	highs   []float64
	lows    []float64
	closes  []float64
	volumes []float64
}

type Preclose14Row struct {
	Date, Ticker string
	Cols         map[string]float64
}

func idxTickSize(price float64) float64 {
	if price <= 0 {
		return 0
	}
	if price < 200 {
		return 1.0
	}
	if price < 500 {
		return 2.0
	}
	if price < 2000 {
		return 5.0
	}
	if price < 5000 {
		return 10.0
	}
	return 25.0
}

func estimateSpreadFrac(price float64) float64 {
	if price <= 0 {
		return 0
	}
	tick := idxTickSize(price)
	var mult float64
	if price < 200 {
		mult = 2.5
	} else if price < 500 {
		mult = 2.0
	} else if price < 5000 {
		mult = 1.5
	} else {
		mult = 1.0
	}
	return (tick * mult) / price
}

func araLimitPct(refPrice float64) float64 {
	if refPrice <= 0 || math.IsNaN(refPrice) {
		return math.NaN()
	}
	if refPrice <= 200 {
		return 0.35
	}
	if refPrice <= 5000 {
		return 0.25
	}
	return 0.20
}

func ComputePreclose14(yf1hPath, targetDate string) ([]Preclose14Row, error) {
	dateFmt := "2006-01-02"
	targetDay, _ := time.Parse(dateFmt, targetDate)
	lookback := targetDay.AddDate(0, 0, -60)

	allBars, err := parquet.ReadFile[p14Bar](yf1hPath)
	if err != nil {
		return nil, err
	}

	// Step 1: Find target tickers and build prev_close
	targetTickers := make(map[string]bool)
	tickerCloses := make(map[string]map[time.Time]float64) // ticker -> date -> last close

	for _, b := range allBars {
		t := strings.TrimSuffix(b.Ticker, ".JK")
		d := b.Date
		if d.IsZero() {
			continue
		}
		day := d.Truncate(24 * time.Hour)

		if b.Close != 0 && !math.IsNaN(b.Close) {
			if tickerCloses[t] == nil {
				tickerCloses[t] = make(map[time.Time]float64)
			}
			tickerCloses[t][day] = b.Close
		}

		hour := jakartaHour(d)
		if day.Equal(targetDay) && p14PrecloseHours[hour] {
			targetTickers[t] = true
		}
	}

	// Step 2: Build prev_close for target tickers
	prevCloseMap := make(map[string]float64)
	for ticker := range targetTickers {
		days := tickerCloses[ticker]
		if days == nil {
			continue
		}
		sorted := make([]time.Time, 0, len(days))
		for d := range days {
			sorted = append(sorted, d)
		}
		sort.Slice(sorted, func(i, j int) bool { return sorted[i].Before(sorted[j]) })
		for i := 1; i < len(sorted); i++ {
			key := sorted[i].Format(dateFmt) + "|" + ticker
			prevCloseMap[key] = days[sorted[i-1]]
		}
	}

	// Step 3: Collect preclose bars for target tickers only
	type keyT struct {
		date   time.Time
		ticker string
	}
	barsByKey := make(map[keyT][]p14Bar)
	for _, b := range allBars {
		t := strings.TrimSuffix(b.Ticker, ".JK")
		if !targetTickers[t] {
			continue
		}
		d := b.Date
		if d.IsZero() {
			continue
		}
		day := d.Truncate(24 * time.Hour)
		if day.Before(lookback) || day.After(targetDay) {
			continue
		}
		hour := jakartaHour(d)
		if !p14PrecloseHours[hour] {
			continue
		}
		key := keyT{day, t}
		barsByKey[key] = append(barsByKey[key], b)
	}

	// Step 4: Group by ticker and sort
	type tickerDay struct {
		date time.Time
		bars []p14Bar
	}
	tickerDays := make(map[string][]tickerDay)
	for key, bars := range barsByKey {
		tickerDays[key.ticker] = append(tickerDays[key.ticker], tickerDay{key.date, bars})
	}

	// Step 5: Build daily aggregates and compute features
	var results []Preclose14Row
	for ticker, tdList := range tickerDays {
		sort.Slice(tdList, func(i, j int) bool { return tdList[i].date.Before(tdList[j].date) })

		var aggs []p14DailyAgg
		for _, td := range tdList {
			a := buildDailyAgg(td)
			key := td.date.Format(dateFmt) + "|" + ticker
			if pc, ok := prevCloseMap[key]; ok {
				a.prevClose = pc
			}
			aggs = append(aggs, a)
		}

		volHist := make([]float64, len(aggs))
		turnHist := make([]float64, len(aggs))
		for i := range aggs {
			volHist[i] = aggs[i].vol
			turnHist[i] = aggs[i].tpVol
		}

		for i, a := range aggs {
			if !a.date.Equal(targetDay) {
				continue
			}
			row := computeFeatures(&a, volHist, turnHist, i)
			row.Date = targetDate
			row.Ticker = ticker
			results = append(results, row)
		}
	}

	return results, nil
}

// Convert to Jakarta timezone (UTC+7) for hour-based filtering
func jakartaHour(t time.Time) int {
	loc, _ := time.LoadLocation("Asia/Jakarta")
	return t.In(loc).Hour()
}

func buildDailyAgg(td struct {
	date time.Time
	bars []p14Bar
}) p14DailyAgg {
	a := p14DailyAgg{
		date:    td.date,
		high:    math.Inf(-1),
		low:     math.Inf(1),
		pmHigh:  math.Inf(-1),
		pmLow:   math.Inf(1),
		amHigh:  math.Inf(-1),
		amLow:   math.Inf(1),
		orbHigh: math.Inf(-1),
		orbLow:  math.Inf(1),
	}
	sort.Slice(td.bars, func(i, j int) bool { return td.bars[i].Date.Before(td.bars[j].Date) })

	var sumVolB14, countB14 float64
	for _, b := range td.bars {
		h := jakartaHour(b.Date)
		o := b.Open
		hi := b.High
		lo := b.Low
		c := b.Close
		v := b.Volume
		if math.IsNaN(o) || math.IsNaN(hi) || math.IsNaN(lo) || math.IsNaN(c) {
			continue
		}
		if math.IsNaN(v) {
			v = 0
		}
		a.hours = append(a.hours, h)
		a.opens = append(a.opens, o)
		a.highs = append(a.highs, hi)
		a.lows = append(a.lows, lo)
		a.closes = append(a.closes, c)
		a.volumes = append(a.volumes, v)

		if hi > a.high {
			a.high = hi
		}
		if lo < a.low {
			a.low = lo
		}
		tp := (hi + lo + c) / 3.0
		a.tpVol += tp * v
		a.vol += v

		if a.open9 == 0 {
			a.open9 = o
		}
		if h == 14 {
			if a.open14 == 0 {
				a.open14 = o
			}
			a.close14 = c
			a.vol14h += v
		}
		if p14MorningHours[h] {
			if a.orbOpen == 0 && a.orbClose == 0 && math.IsInf(a.orbHigh, -1) {
				a.orbOpen = o
				a.orbClose = c
				a.orbHour = h
			}
			if hi > a.orbHigh {
				a.orbHigh = hi
			}
			if lo < a.orbLow {
				a.orbLow = lo
			}
			a.orbClose = c
			if hi > a.amHigh {
				a.amHigh = hi
			}
			if lo < a.amLow {
				a.amLow = lo
			}
		}
		if h == 11 {
			a.close11 = c
		}
		if h == 13 || h == 14 {
			if hi > a.pmHigh {
				a.pmHigh = hi
			}
			if lo < a.pmLow {
				a.pmLow = lo
			}
		}
		if h >= 13 && a.openPM == 0 {
			a.openPM = o
			a.closePM = c
		}
		if p14MorningHours[h] {
			a.amTpVol += tp * v
			a.amVol += v
		}
		if h >= 9 && h <= 13 {
			sumVolB14 += v
			countB14++
		}
	}
	if countB14 > 0 {
		a.avgHourly = sumVolB14 / countB14
	}
	zeroIfInf(&a.high)
	zeroIfInf(&a.low)
	zeroIfInf(&a.pmHigh)
	zeroIfInf(&a.pmLow)
	zeroIfInf(&a.amHigh)
	zeroIfInf(&a.amLow)
	zeroIfInf(&a.orbHigh)
	zeroIfInf(&a.orbLow)
	return a
}

func zeroIfInf(v *float64) {
	if math.IsInf(*v, -1) || math.IsInf(*v, 1) {
		*v = 0
	}
}

func computeFeatures(a *p14DailyAgg, volHist, turnHist []float64, idx int) Preclose14Row {
	r := Preclose14Row{Cols: make(map[string]float64)}

	set := func(k string, v float64) {
		if !math.IsNaN(v) && !math.IsInf(v, 0) {
			r.Cols[k] = v
		}
	}

	set("pre14_open9", a.open9)
	set("pre14_high", a.high)
	set("pre14_low", a.low)
	set("pre14_tp_vol", a.tpVol)
	set("pre14_volume_until14", a.vol)
	set("pre14_open14", a.open14)
	set("pre14_close14", a.close14)
	set("pre14_volume_14h", a.vol14h)
	set("pre14_orb_open", a.orbOpen)
	set("pre14_orb_high", a.orbHigh)
	set("pre14_orb_low", a.orbLow)
	set("pre14_orb_close", a.orbClose)
	set("pre14_orb_source_hour", float64(a.orbHour))
	set("pre14_close11", a.close11)
	set("pre14_am_high", a.amHigh)
	set("pre14_am_low", a.amLow)
	set("pre14_pm_high", a.pmHigh)
	set("pre14_pm_low", a.pmLow)
	set("pre14_open_pm", a.openPM)
	set("pre14_close_pm_open", a.closePM)
	set("pre14_am_volume", a.amVol)
	set("pre14_pm_volume", a.vol-a.amVol)
	set("pre14_am_turnover", a.amTpVol)
	set("pre14_pm_turnover", a.tpVol-a.amTpVol)

	if a.amVol > 0 {
		set("pre14_pm_am_volume_ratio", (a.vol-a.amVol)/a.amVol)
		set("pre14_pm_am_volume_rate_ratio", ((a.vol-a.amVol)/2.0)/(a.amVol/3.0))
	}
	if a.amTpVol > 0 {
		set("pre14_pm_am_turnover_ratio", (a.tpVol-a.amTpVol)/a.amTpVol)
		set("pre14_pm_am_turnover_rate_ratio", ((a.tpVol-a.amTpVol)/2.0)/(a.amTpVol/3.0))
	}

	if a.open14 > 0 {
		set("pre14_ret_14h", (a.close14-a.open14)/a.open14)
	}
	if a.close11 > 0 {
		set("pre14_ret_11_to_14", (a.close14-a.close11)/a.close11)
		set("pre14_pm_high_vs_close11", (a.pmHigh-a.close11)/a.close11)
		set("pre14_pm_close_vs_close11", (a.close14-a.close11)/a.close11)
		set("pre14_pm_low_vs_close11", (a.pmLow-a.close11)/a.close11)
		set("pre14_pm_open_vs_close11", (a.openPM-a.close11)/a.close11)
		set("pre14_pm_high_close_spread", (a.pmHigh-a.close14)/a.close11)
	}
	if a.open9 > 0 {
		set("pre14_close_vs_open9", (a.close14-a.open9)/a.open9)
		set("pre14_range_pct", (a.high-a.low)/a.open9)
	}

	if a.vol > 0 {
		vwap := a.tpVol / a.vol
		set("pre14_vwap", vwap)
		if vwap > 0 {
			set("pre14_close_to_vwap", (a.close14-vwap)/vwap)
			if a.closePM > 0 {
				set("pre14_pm_drive", (a.close14-a.closePM)/vwap)
			}
		}
		set("pre14_close_above_vwap", p14Bool(a.close14 > vwap))
	}
	if a.amVol > 0 {
		amVwap := a.amTpVol / a.amVol
		set("pre14_vwap_morning", amVwap)
		if amVwap > 0 {
			if a.vol > 0 {
				vwap := a.tpVol / a.vol
				set("pre14_vwap_trend", (vwap-amVwap)/amVwap)
			}
			if a.closePM > 0 {
				set("pre14_pm_open_to_vwap_am", (a.closePM-amVwap)/amVwap)
			}
		}
	}
	set("pre14_turnover_until14", a.tpVol)

	orbMid := (a.orbHigh + a.orbLow) / 2.0
	orbRange := a.orbHigh - a.orbLow
	set("pre14_orb_mid", orbMid)
	set("pre14_orb_range", orbRange)
	if a.orbOpen > 0 {
		set("pre14_orb_range_pct", orbRange/a.orbOpen)
		set("pre14_orb_body_pct", (a.orbClose-a.orbOpen)/a.orbOpen)
		set("pre14_orb_upper_wick_pct", (a.orbHigh-math.Max(a.orbOpen, a.orbClose))/a.orbOpen)
		set("pre14_orb_lower_wick_pct", (math.Min(a.orbOpen, a.orbClose)-a.orbLow)/a.orbOpen)
	}
	if orbMid > 0 {
		set("pre14_close_vs_orb_mid", (a.close14-orbMid)/orbMid)
	}
	if a.orbHigh > 0 {
		set("pre14_close_vs_orb_high", (a.close14-a.orbHigh)/a.orbHigh)
		set("pre14_pm_high_vs_orb_high", (a.pmHigh-a.orbHigh)/a.orbHigh)
	}
	if a.orbLow > 0 {
		set("pre14_close_vs_orb_low", (a.close14-a.orbLow)/a.orbLow)
	}
	if orbRange > 0 {
		set("pre14_orb_position", (a.close14-a.orbLow)/orbRange)
		set("pre14_orb_breakout_strength", p14Clip(a.close14-a.orbHigh)/orbRange)
		set("pre14_orb_breakdown_strength", p14Clip(a.orbLow-a.close14)/orbRange)
		set("pre14_pm_high_breakout_strength", p14Clip(a.pmHigh-a.orbHigh)/orbRange)
		set("pre14_close_orb_high_hold", (a.close14-a.orbHigh)/orbRange)
	}
	set("pre14_close_above_orb_high", p14Bool(a.close14 > a.orbHigh))
	set("pre14_close_below_orb_low", p14Bool(a.close14 < a.orbLow))
	set("pre14_pm_high_above_orb_high", p14Bool(a.pmHigh > a.orbHigh))

	cumVol := 0.0
	cumTpVol := 0.0
	volAboveVwap := 0.0
	for j := 0; j < len(a.hours); j++ {
		cumVol += a.volumes[j]
		cumTpVol += (a.highs[j] + a.lows[j] + a.closes[j]) / 3.0 * a.volumes[j]
		if cumVol > 0 && a.closes[j] > cumTpVol/cumVol {
			volAboveVwap += a.volumes[j]
		}
	}
	if a.vol > 0 {
		set("pre14_vol_above_vwap_pct", volAboveVwap/a.vol)
	}

	for _, w := range []int{5, 10, 20} {
		volMA := rollingMAShifted(volHist, idx, w)
		turnMA := rollingMAShifted(turnHist, idx, w)
		set(fmt.Sprintf("pre14_daily_volume_ma%d", w), volMA)
		set(fmt.Sprintf("pre14_daily_turnover_ma%d", w), turnMA)
		if volMA > 0 {
			set(fmt.Sprintf("pre14_am_volume_to_dailyvol_%dd", w), a.amVol/volMA)
			set(fmt.Sprintf("pre14_until14_volume_to_dailyvol_%dd", w), a.vol/volMA)
		}
		if turnMA > 0 {
			set(fmt.Sprintf("pre14_am_turnover_to_dailyturnover_%dd", w), a.amTpVol/turnMA)
			set(fmt.Sprintf("pre14_until14_turnover_to_dailyturnover_%dd", w), a.tpVol/turnMA)
		}
	}
	volMA20 := rollingMAShifted(volHist, idx, 20)
	turnMA20 := rollingMAShifted(turnHist, idx, 20)
	if volMA20 > 0 {
		set("pre14_volume_until14_ratio_20d", a.vol/volMA20)
	}
	if turnMA20 > 0 {
		set("pre14_turnover_until14_ratio_20d", a.tpVol/turnMA20)
	}
	if a.avgHourly > 0 {
		set("pre14_volume_14h_ratio_to_avg_hourly", a.vol14h/a.avgHourly)
	}

	ts := idxTickSize(a.close14)
	set("pre14_tick_size", ts)
	if a.close14 > 0 {
		set("pre14_tick_pct", ts/a.close14)
	}
	sp := estimateSpreadFrac(a.close14)
	set("pre14_spread_cost_est", 2.0*sp)
	set("pre14_market_cost_est", p14RoundtripCostFrac+2.0*sp)

	if a.open9 > 0 {
		set("pre14_am_open_low_dist", (a.open9-a.amLow)/a.open9)
	}
	if a.openPM > 0 {
		set("pre14_pm_open_low_dist", (a.openPM-a.pmLow)/a.openPM)
	}
	set("pre14_am_open_is_low", p14Bool(a.open9 <= a.amLow+ts))
	set("pre14_pm_open_is_low", p14Bool(a.openPM <= a.pmLow+ts))

	if pmRange := a.pmHigh - a.pmLow; pmRange > 0 {
		v := (a.close14 - a.pmLow) / pmRange
		if v < 0 {
			v = 0
		}
		if v > 1 {
			v = 1
		}
		set("pre14_pm_close_near_high", v)
	}

	set("pre14_prev_close", a.prevClose)
	if a.prevClose > 0 {
		rpc := (a.close14 - a.prevClose) / a.prevClose
		set("pre14_return_from_prev_close", rpc)
		al := araLimitPct(a.prevClose)
		set("pre14_ara_limit_pct", al)
		set("pre14_ara_distance_pct", al-rpc)
		set("pre14_is_ara_like", p14Bool(rpc >= al-p14AraBufferPct))
		araPrice := a.prevClose * (1.0 + al)
		set("pre14_ara_price", araPrice)
		computeARAState(&r, a, araPrice, ts)
	}
	return r
}

func computeARAState(r *Preclose14Row, a *p14DailyAgg, araPrice, tickSize float64) {
	if araPrice <= 0 || len(a.hours) == 0 {
		return
	}
	var touchCnt, releaseCnt, closeAraCnt, flatLockedCnt float64
	var depths []float64
	var firstH int
	var lbT, lbR, lbCA, lbFL, lbRD float64
	n := len(a.hours)

	for i := 0; i < n; i++ {
		hi := a.highs[i]
		lo := a.lows[i]
		cl := a.closes[i]
		op := a.opens[i]
		t := hi >= (araPrice - tickSize)
		rel := t && lo < (araPrice-tickSize)
		cAra := cl >= (araPrice - tickSize)
		fl := t && op >= (araPrice-tickSize) && lo >= (araPrice-tickSize) && cl >= (araPrice-tickSize)
		last := i == n-1
		if t {
			touchCnt++
			if firstH == 0 {
				firstH = a.hours[i]
			}
		}
		if rel {
			releaseCnt++
			depths = append(depths, (araPrice-lo)/araPrice)
		}
		if cAra {
			closeAraCnt++
		}
		if fl {
			flatLockedCnt++
		}
		if last {
			lbT = p14Bool(t)
			lbR = p14Bool(rel)
			lbCA = p14Bool(cAra)
			lbFL = p14Bool(fl)
			if rel {
				lbRD = (araPrice - lo) / araPrice
			}
		}
	}
	mxD := 0.0
	sumD := 0.0
	for _, d := range depths {
		sumD += d
		if d > mxD {
			mxD = d
		}
	}
	mnD := 0.0
	if len(depths) > 0 {
		mnD = sumD / float64(len(depths))
	}

	set := func(k string, v float64) {
		if !math.IsNaN(v) && !math.IsInf(v, 0) {
			r.Cols[k] = v
		}
	}
	set("pre14_ara_touched", p14Bool(touchCnt > 0))
	set("pre14_ara_touch_hour", float64(firstH))
	set("pre14_ara_touched_bar_count", touchCnt)
	set("pre14_ara_release_wick_count", releaseCnt)
	if touchCnt > 0 {
		set("pre14_ara_release_wick_ratio", releaseCnt/touchCnt)
	}
	set("pre14_ara_release_wick_depth_max", mxD)
	set("pre14_ara_release_wick_depth_mean", mnD)
	set("pre14_ara_close_at_ara_count", closeAraCnt)
	set("pre14_ara_flat_ohlc_count", flatLockedCnt)
	set("pre14_ara_last_bar_touched", lbT)
	set("pre14_ara_last_bar_release", lbR)
	set("pre14_ara_last_bar_close_at_ara", lbCA)
	set("pre14_ara_last_bar_locked", lbFL)
	set("pre14_ara_last_bar_release_depth", lbRD)
	set("pre14_ara_locked_proxy", p14Bool(touchCnt > 0 && releaseCnt <= 0 && lbCA > 0))
	set("pre14_ara_touched_released", p14Bool(touchCnt > 0 && releaseCnt > 0))
}

func rollingMAShifted(hist []float64, i, window int) float64 {
	start := i - window
	if start < 0 {
		start = 0
	}
	sl := hist[start:i]
	mp := max(3, window/4)
	if len(sl) < mp {
		return 0
	}
	sum := 0.0
	for _, x := range sl {
		if math.IsNaN(x) {
			continue
		}
		sum += x
	}
	return sum / float64(len(sl))
}

func p14Clip(x float64) float64 {
	if x < 0 {
		return 0
	}
	return x
}

func p14Bool(b bool) float64 {
	if b {
		return 1.0
	}
	return 0.0
}
