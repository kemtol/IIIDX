// probe_trend — minimal IPOT WebSocket probe for TREND_1D subscription.
//
// Goal: verify whether the candidate WS subscription endpoint is valid,
// and inspect the shape of frames it returns. No persistence yet.
//
// Usage:
//   go run . --code DKHH --seconds 30
package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"net/http"
	"os"
	"time"

	"nhooyr.io/websocket"
)

const (
	wsURL  = "wss://ipotapp.ipot.id/socketcluster/?appsession="
	origin = "https://indopremier.com"
)

func main() {
	code := flag.String("code", "DKHH", "ticker code to subscribe")
	seconds := flag.Int("seconds", 30, "how long to listen, in seconds")
	out := flag.String("out", "/tmp/probe_trend_frames.jsonl", "output JSONL path")
	rtype := flag.String("rtype", "TREND_1D", "rtype to subscribe")
	subsid := flag.String("subsid", "stockdashboard_trending", "subsid")
	flag.Parse()

	f, err := os.Create(*out)
	if err != nil {
		fail("create output: %v", err)
	}
	defer f.Close()

	ctx, cancel := context.WithTimeout(context.Background(), time.Duration(*seconds+10)*time.Second)
	defer cancel()

	logf("connecting to %s", wsURL)
	conn, _, err := websocket.Dial(ctx, wsURL, &websocket.DialOptions{
		HTTPHeader: http.Header{
			"Origin":     []string{origin},
			"User-Agent": []string{"probe-trend/0.1"},
		},
		CompressionMode: websocket.CompressionDisabled,
	})
	if err != nil {
		fail("dial: %v", err)
	}
	defer conn.Close(websocket.StatusNormalClosure, "bye")

	// 1) handshake
	handshake := map[string]any{
		"event": "#handshake",
		"data":  map[string]any{"authToken": nil},
		"cid":   1,
	}
	if err := writeJSON(ctx, conn, handshake); err != nil {
		fail("handshake send: %v", err)
	}
	logf("sent handshake")

	// 2) subscribe — exact shape requested by user
	subscribe := map[string]any{
		"event": "cmd",
		"data": map[string]any{
			"cmdid": 17,
			"param": map[string]any{
				"cmd":       "subscribe",
				"service":   "mi",
				"rtype":     *rtype,
				"subsid":    *subsid,
				"code":      *code,
				"subscribe": true,
			},
		},
		"cid": 19,
	}
	if err := writeJSON(ctx, conn, subscribe); err != nil {
		fail("subscribe send: %v", err)
	}
	logf("sent subscribe rtype=%s code=%s", *rtype, *code)

	// 3) read loop
	deadline := time.Now().Add(time.Duration(*seconds) * time.Second)
	frameCount := 0
	postHandshakeCount := 0

	for time.Now().Before(deadline) {
		readCtx, readCancel := context.WithDeadline(ctx, deadline)
		_, raw, err := conn.Read(readCtx)
		readCancel()
		if err != nil {
			if ctx.Err() != nil {
				logf("ctx done: %v", ctx.Err())
				break
			}
			logf("read error: %v", err)
			break
		}

		frameCount++

		// pretty-print to stdout, raw line to JSONL
		var pretty any
		if err := json.Unmarshal(raw, &pretty); err == nil {
			b, _ := json.MarshalIndent(pretty, "", "  ")
			fmt.Printf("--- frame %d ---\n%s\n", frameCount, string(b))
			// Heuristic: anything not a pure handshake response is "interesting".
			if m, ok := pretty.(map[string]any); ok {
				if _, hasRid := m["rid"]; hasRid {
					if cid, _ := m["cid"]; cid == nil {
						postHandshakeCount++
					}
				}
				if ev, _ := m["event"].(string); ev != "" && ev != "#handshake" {
					postHandshakeCount++
				}
			}
		} else {
			fmt.Printf("--- frame %d (non-json) ---\n%s\n", frameCount, string(raw))
		}

		f.Write(raw)
		f.Write([]byte("\n"))
	}

	// 4) verdict
	logf("done. total=%d frames, post-handshake-ish=%d", frameCount, postHandshakeCount)
	if frameCount == 0 {
		logf("VERDICT: NO frames received — likely auth required or endpoint invalid")
		os.Exit(2)
	}
	if postHandshakeCount == 0 {
		logf("VERDICT: only handshake-like frames — subscribe may have been rejected silently")
		os.Exit(3)
	}
	logf("VERDICT: at least one non-handshake frame — endpoint likely valid, inspect %s", *out)
}

func writeJSON(ctx context.Context, conn *websocket.Conn, v any) error {
	b, err := json.Marshal(v)
	if err != nil {
		return err
	}
	return conn.Write(ctx, websocket.MessageText, b)
}

func logf(format string, args ...any) {
	fmt.Fprintf(os.Stderr, "[probe] "+format+"\n", args...)
}

func fail(format string, args ...any) {
	fmt.Fprintf(os.Stderr, "[probe FATAL] "+format+"\n", args...)
	os.Exit(1)
}
