// probe_trend — Concurrent IPOT WebSocket scraper
package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"net/http"
	"os"
	"strings"
	"sync"
	"time"

	"nhooyr.io/websocket"
)

const (
	wsURL  = "wss://ipotapp.ipot.id/socketcluster/?appsession="
	origin = "https://indopremier.com"
)

type Frame struct {
	Event string `json:"event"`
	Rid   int    `json:"rid,omitempty"`
	Data  struct {
		Code string `json:"code"`
		Data struct {
			Index int `json:"index"`
			Data  map[string]any `json:"data"`
		} `json:"data"`
		RType string `json:"rtype"`
	} `json:"data"`
}

func main() {
	tickerList := flag.String("tickers", "DKHH,GOTO,BBCA,BBRI,BMRI,TLKM,ASII,BBNI,UNTR,ADRO,PTBA,ITMG,AKRA,AMRT,BRIS,ANTM,INCO,MEDC,PGAS,CPIN", "comma separated tickers")
	seconds := flag.Int("seconds", 15, "how long to listen")
	out := flag.String("out", "trending_multi.jsonl", "output JSONL")
	flag.Parse()

	tickers := strings.Split(*tickerList, ",")
	f, _ := os.Create(*out)
	defer f.Close()

	ctx, cancel := context.WithTimeout(context.Background(), time.Duration(*seconds+5)*time.Second)
	defer cancel()

	conn, _, err := websocket.Dial(ctx, wsURL, &websocket.DialOptions{
		HTTPHeader: http.Header{"Origin": []string{origin}},
	})
	if err != nil {
		fmt.Printf("Dial error: %v\n", err)
		return
	}
	defer conn.Close(websocket.StatusNormalClosure, "")

	// Handshake
	writeJSON(ctx, conn, map[string]any{"event": "#handshake", "data": map[string]any{"authToken": nil}, "cid": 1})

	// Subscribe to all tickers concurrently (send commands)
	start := time.Now()
	for i, code := range tickers {
		subscribe := map[string]any{
			"event": "cmd",
			"data": map[string]any{
				"cmdid": 17,
				"param": map[string]any{
					"cmd": "subscribe", "service": "mi", "rtype": "TREND_1D", "subsid": "trending", "code": code, "subscribe": true,
				},
			},
			"cid": i + 10,
		}
		writeJSON(ctx, conn, subscribe)
	}

	fmt.Printf("[Go] Subscribed to %d tickers in %v\n", len(tickers), time.Since(start))

	// Read loop
	var wg sync.WaitGroup
	wg.Add(1)
	
	frameCount := 0
	go func() {
		defer wg.Done()
		for {
			_, raw, err := conn.Read(ctx)
			if err != nil {
				return
			}
			f.Write(raw)
			f.Write([]byte("\n"))
			frameCount++
		}
	}()

	time.Sleep(time.Duration(*seconds) * time.Second)
	cancel()
	wg.Wait()

	fmt.Printf("[Go] Finished. Total frames captured: %d in %d seconds\n", frameCount, *seconds)
}

func writeJSON(ctx context.Context, conn *websocket.Conn, v any) error {
	b, _ := json.Marshal(v)
	return conn.Write(ctx, websocket.MessageText, b)
}
