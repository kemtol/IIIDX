package notify

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"time"
)

// CheckItem is a single preflight check result.
type CheckItem struct {
	Name, Status, Detail string
}

type discordMessage struct {
	Embeds []discordEmbed `json:"embeds,omitempty"`
	Content string        `json:"content,omitempty"`
}

type discordEmbed struct {
	Title       string              `json:"title,omitempty"`
	Description string              `json:"description,omitempty"`
	Color       int                 `json:"color"`
	Fields      []discordField      `json:"fields,omitempty"`
	Footer      *discordFooter      `json:"footer,omitempty"`
}

type discordField struct {
	Name   string `json:"name"`
	Value  string `json:"value"`
	Inline bool   `json:"inline"`
}

type discordFooter struct {
	Text string `json:"text"`
}

// SendDiscord sends a plain text message to a Discord channel via bot token.
func SendDiscord(token, channelID, text string) error {
	url := fmt.Sprintf("https://discord.com/api/v10/channels/%s/messages", channelID)

	msg := discordMessage{Content: text}
	body, err := json.Marshal(msg)
	if err != nil {
		return fmt.Errorf("marshal: %w", err)
	}

	req, err := http.NewRequest("POST", url, bytes.NewReader(body))
	if err != nil {
		return fmt.Errorf("request: %w", err)
	}
	req.Header.Set("Authorization", "Bot "+token)
	req.Header.Set("Content-Type", "application/json")

	client := &http.Client{Timeout: 10 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		return fmt.Errorf("post: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		return fmt.Errorf("discord returned %d", resp.StatusCode)
	}
	return nil
}

// SendDiscordPreflight sends a preflight checklist as a Discord embed.
func SendDiscordPreflight(token, channelID, title string, items []CheckItem, allOK bool) error {
	url := fmt.Sprintf("https://discord.com/api/v10/channels/%s/messages", channelID)

	desc := ""
	for _, it := range items {
		desc += fmt.Sprintf("%s **%s**", it.Status, it.Name)
		if it.Detail != "" {
			desc += fmt.Sprintf("  `%s`", it.Detail)
		}
		desc += "\n"
	}

	color := 0xFFA500 // orange
	footerText := "Rechecking…"
	if allOK {
		color = 0x00FF00 // green
		footerText = "Ready for inference"
	}

	embeds := []discordEmbed{{
		Title:       title,
		Description: desc,
		Color:       color,
		Footer:      &discordFooter{Text: footerText},
	}}

	msg := discordMessage{Embeds: embeds}
	body, err := json.Marshal(msg)
	if err != nil {
		return err
	}

	req, err := http.NewRequest("POST", url, bytes.NewReader(body))
	if err != nil {
		return err
	}
	req.Header.Set("Authorization", "Bot "+token)
	req.Header.Set("Content-Type", "application/json")

	client := &http.Client{Timeout: 10 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		return fmt.Errorf("discord returned %d", resp.StatusCode)
	}
	return nil
}
