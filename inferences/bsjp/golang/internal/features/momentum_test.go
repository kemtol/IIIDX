package features

import (
	"testing"
	"time"
)

func TestJakartaDateHourUsesWIB(t *testing.T) {
	ts := time.Date(2026, 5, 7, 8, 18, 56, 0, time.UTC)

	date, hour := jakartaDateHour(ts)

	if date != "2026-05-07" {
		t.Fatalf("date = %q, want 2026-05-07", date)
	}
	if hour != 15 {
		t.Fatalf("hour = %d, want 15", hour)
	}
}
