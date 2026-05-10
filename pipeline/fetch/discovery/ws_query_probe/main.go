package main

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"time"

	"nhooyr.io/websocket"
)

const (
	wsURL  = "wss://ipotapp.ipot.id/socketcluster/?appsession="
	origin = "https://indopremier.com"
)

func main() {
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	conn, _, err := websocket.Dial(ctx, wsURL, &websocket.DialOptions{
		HTTPHeader: http.Header{
			"Origin":     []string{origin},
			"User-Agent": []string{"broksum-scrapper/1.0"},
		},
	})
	if err != nil {
		fmt.Printf("Dial error: %v\n", err)
		return
	}
	defer conn.Close(websocket.StatusNormalClosure, "")

	// 1. Handshake
	writeJSON(ctx, conn, map[string]any{"event": "#handshake", "data": map[string]any{"authToken": nil}, "cid": 1})

	// 2. Commands (Note: the dates MUST be valid trading days)
	commands := []string{
		`{"event":"cmd","data":{"cmdid":41,"param":{"service":"midata","cmd":"query","param":{"source":"datafeed","index":"en_qu_top_bs","args":["s","TLKM","","%","%","2026-05-07","2026-05-07"]}}},"cid":43}`,
		`{"event":"cmd","data":{"cmdid":42,"param":{"service":"midata","cmd":"query","param":{"source":"datafeed","index":"en_qu_top_bs","args":["s","TLKM","","%","F","2026-05-07","2026-05-07"]}}},"cid":44}`,
		`{"event":"cmd","data":{"cmdid":43,"param":{"cmd":"query","service":"midata","param":{"source":"datafeed","index":"xen_qu_stock_gl","args":[["TLKM"],"2026-05-07","2026-05-07"]}}},"cid":45}`,
	}

	fmt.Println("[Go] Waiting for handshake...")
	for i := 0; i < 100; i++ {
		_, raw, err := conn.Read(ctx)
		if err != nil {
			break
		}
		
		var msg map[string]any
		json.Unmarshal(raw, &msg)
		
		event, _ := msg["event"].(string)
		rid, _ := msg["rid"]

		if rid == float64(1) {
			fmt.Println("[Go] Handshake OK. Sending commands...")
			for _, cmdStr := range commands {
				var cmd map[string]any
				json.Unmarshal([]byte(cmdStr), &cmd)
				writeJSON(ctx, conn, cmd)
			}
		}

		// TANGKAP EVENT "record"
		if event == "record" {
			fmt.Printf("\n--- RECORD RECEIVED ---\n")
			pretty, _ := json.MarshalIndent(msg, "", "  ")
			fmt.Println(string(pretty))
		} else if rid != nil && rid.(float64) > 1 {
			fmt.Printf("[Go] Command response for RID %v: %v\n", rid, msg["data"])
		}
	}
}

func writeJSON(ctx context.Context, conn *websocket.Conn, v any) error {
	b, _ := json.Marshal(v)
	return conn.Write(ctx, websocket.MessageText, b)
}
