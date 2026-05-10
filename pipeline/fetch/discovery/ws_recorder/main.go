// ws_recorder — Persistent market-wide WebSocket recorder
package main

import (
	"context"
	"flag"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/signal"
	"regexp"
	"strings"
	"sync"
	"syscall"
	"time"

	"nhooyr.io/websocket"
)

const (
	appSessionURL = "https://indopremier.com/ipc/appsession.js"
	wsBaseURL     = "wss://ipotapp.ipot.id/socketcluster/"
	origin        = "https://indopremier.com"
	userAgent     = "broksum-scrapper/1.0"
)

func getAppSession() (string, error) {
	resp, err := http.Get(appSessionURL)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(resp.Body)
	re := regexp.MustCompile(`appsession\s*[:=]\s*["']([^"']+)["']`)
	matches := re.FindStringSubmatch(string(body))
	if len(matches) > 1 {
		return matches[1], nil
	}
	return "", fmt.Errorf("appsession not found")
}

func main() {
	tickerFile := flag.String("ticker-file", "../../../../data/Level_0_Raw/all_tickers_clean.txt", "path to tickers file")
	outDir := flag.String("out-dir", "../../../../data/Level_0_Raw/ws_captures", "output directory")
	flag.Parse()

	os.MkdirAll(*outDir, 0755)
	
	data, err := os.ReadFile(*tickerFile)
	if err != nil {
		fmt.Printf("Fatal: could not read ticker file: %v\n", err)
		return
	}
	tickers := strings.Split(string(data), ",")

	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)

	for {
		ctx, cancel := context.WithCancel(context.Background())
		
		fmt.Println("[Recorder] Starting new session...")
		runSession(ctx, tickers, *outDir)
		
		select {
		case <-sigChan:
			fmt.Println("[Recorder] Shutting down...")
			cancel()
			return
		case <-time.After(5 * time.Second):
			fmt.Println("[Recorder] Session ended or crashed. Reconnecting in 5s...")
			cancel()
		}
	}
}

func runSession(ctx context.Context, tickers []string, outDir string) {
	appSession, err := getAppSession()
	if err != nil {
		fmt.Printf("Error: %v\n", err)
		return
	}

	wsURL := wsBaseURL + "?appsession=" + appSession
	conn, _, err := websocket.Dial(ctx, wsURL, &websocket.DialOptions{
		HTTPHeader: http.Header{"Origin": []string{origin}, "User-Agent": []string{userAgent}},
	})
	if err != nil {
		fmt.Printf("Dial error: %v\n", err)
		return
	}
	defer conn.Close(websocket.StatusNormalClosure, "")

	// Handshake
	b, _ := (&struct {
		Event string `json:"event"`
		Data  map[string]any `json:"data"`
		Cid   int `json:"cid"`
	}{Event: "#handshake", Data: map[string]any{"authToken": nil}, Cid: 1}).MarshalJSON()
	conn.Write(ctx, websocket.MessageText, b)

	// Output file (dated)
	fileName := fmt.Sprintf("%s/ws_trend_%s.jsonl", outDir, time.Now().Format("20060102"))
	f, _ := os.OpenFile(fileName, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	defer f.Close()

	fmt.Printf("[Recorder] Connected. Recording to %s\n", fileName)

	var wg sync.WaitGroup
	wg.Add(1)

	// Reader Loop
	go func() {
		defer wg.Done()
		lastHeartbeat := time.Now()
		count := 0
		for {
			_, raw, err := conn.Read(ctx)
			if err != nil {
				return
			}
			f.Write(raw)
			f.Write([]byte("\n"))
			count++

			if time.Since(lastHeartbeat) > 5*time.Minute {
				fmt.Printf("[Heartbeat] %s: Captured %d frames in last 5m\n", time.Now().Format("15:04"), count)
				count = 0
				lastHeartbeat = time.Now()
			}
		}
	}()

	// Wait for RID 1 then subscribe
	time.Sleep(2 * time.Second)
	fmt.Printf("[Recorder] Subscribing to %d tickers...\n", len(tickers))
	for i, code := range tickers {
		sub := map[string]any{
			"event": "cmd",
			"data": map[string]any{
				"cmdid": 17,
				"param": map[string]any{
					"cmd": "subscribe", "service": "mi", "rtype": "TREND_1D", "subsid": "trending", "code": code, "subscribe": true,
				},
			},
			"cid": i + 10,
		}
		jsonSub, _ := (struct {
			Event string `json:"event"`
			Data  map[string]any `json:"data"`
			Cid   int `json:"cid"`
		}{Event: "cmd", Data: sub["data"].(map[string]any), Cid: i + 10}).MarshalJSON()
		// Wait slightly between batches to avoid flooding
		if i % 100 == 0 { time.Sleep(100 * time.Millisecond) }
		conn.Write(ctx, websocket.MessageText, jsonSub)
	}

	<-ctx.Done()
	wg.Wait()
}

// MarshalJSON helper for structured frames
func (s *struct {
	Event string `json:"event"`
	Data  map[string]any `json:"data"`
	Cid   int `json:"cid"`
}) MarshalJSON() ([]byte, error) {
	type Alias struct {
		Event string `json:"event"`
		Data  map[string]any `json:"data"`
		Cid   int `json:"cid"`
	}
	return json.Marshal((*Alias)(s))
}
