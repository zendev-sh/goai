package httpc

import (
	"encoding/base64"
	"math/rand/v2"
	"strings"
	"testing"
)

func decodeStringAccepts(data string) bool {
	_, err := base64.StdEncoding.DecodeString(data)
	return err == nil
}

func TestValidStdBase64MatchesDecodeString(t *testing.T) {
	quantum := strings.Repeat("QUJD", base64ValidationChunk/4)
	cases := []string{
		"", "\n", "\r\n", "QQ==", "QUI=", "QUJD", "Q", "QQ", "QUJ", "QQ=", "Q===", "====",
		"QQ==\n", "QQ==\r\nQQ==", "QQ==QUJD", "QU\nJD", "QUJD\n\n\nQUJD", "QQ=\n=", "@@@@",
		quantum,          // exactly one full chunk
		quantum + "QQ==", // padding in the second chunk
		quantum[:len(quantum)-4] + "QQ==" + "QUJD", // padding ends the first chunk, data follows
		quantum[:len(quantum)-4] + "QQ==" + "\r\n", // padding ends the first chunk, only line breaks follow
		quantum[:len(quantum)-1],                   // incomplete final quantum
		strings.Repeat("\n", 2*base64ValidationChunk) + "QUJD",
	}
	rng := rand.New(rand.NewPCG(1, 2))
	for range 2000 {
		raw := make([]byte, rng.IntN(3*base64ValidationChunk))
		for i := range raw {
			raw[i] = byte(rng.Uint32())
		}
		encoded := base64.StdEncoding.EncodeToString(raw)
		cases = append(cases, encoded, insertLineBreaks(rng, encoded), corruptOneByte(rng, encoded))
	}
	const alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=\r\n-_ @"
	for range 5000 {
		symbols := make([]byte, rng.IntN(24))
		for i := range symbols {
			symbols[i] = alphabet[rng.IntN(len(alphabet))]
		}
		cases = append(cases, string(symbols), quantum[:len(quantum)-4]+string(symbols))
	}
	for _, data := range cases {
		if got, want := validStdBase64(data), decodeStringAccepts(data); got != want {
			t.Fatalf("validStdBase64(%q...) = %v, DecodeString accepts = %v (len %d)", data[:min(len(data), 40)], got, want, len(data))
		}
	}
}

func TestValidStdBase64DoesNotAllocate(t *testing.T) {
	raw := make([]byte, 1<<20)
	for i := range raw {
		raw[i] = byte(i)
	}
	data := base64.StdEncoding.EncodeToString(raw)
	if allocs := testing.AllocsPerRun(10, func() {
		if !validStdBase64(data) {
			t.Fatal("valid image payload rejected")
		}
	}); allocs != 0 {
		t.Fatalf("validStdBase64 allocated %v times per run, want 0", allocs)
	}
}

func insertLineBreaks(rng *rand.Rand, s string) string {
	var b strings.Builder
	for i := range len(s) {
		if rng.IntN(64) == 0 {
			b.WriteString([]string{"\n", "\r\n", "\r"}[rng.IntN(3)])
		}
		b.WriteByte(s[i])
	}
	return b.String()
}

func corruptOneByte(rng *rand.Rand, s string) string {
	if s == "" {
		return "="
	}
	b := []byte(s)
	b[rng.IntN(len(b))] = "=@\n-"[rng.IntN(4)]
	return string(b)
}
