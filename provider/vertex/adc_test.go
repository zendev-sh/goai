package vertex

import (
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/zendev-sh/goai/provider"
)

// --- BF.9 Item 1: ADC TokenSource tests ---

func TestADCTokenSource_Type(t *testing.T) {
	// Without valid credentials, ADCTokenSource may return an error.
	// If it succeeds, verify the interface.
	ts, err := ADCTokenSource(t.Context())
	if err != nil {
		// Expected when no GCP credentials are configured.
		return
	}
	if ts == nil {
		t.Fatal("ADCTokenSource returned nil without error")
	}
	var _ provider.TokenSource = ts
}

func TestADCTokenSource_FindCredsError(t *testing.T) {
	// Point GOOGLE_APPLICATION_CREDENTIALS at a nonexistent file to force
	// google.FindDefaultCredentials to fail -- exercises the creds error path.
	t.Setenv("GOOGLE_APPLICATION_CREDENTIALS", "/nonexistent/service-account.json")
	t.Setenv("GOOGLE_API_KEY", "")
	t.Setenv("GEMINI_API_KEY", "")
	t.Setenv("GOOGLE_GENERATIVE_AI_API_KEY", "")

	_, err := ADCTokenSource(t.Context())
	if err == nil {
		t.Fatal("expected error when GOOGLE_APPLICATION_CREDENTIALS points to nonexistent file")
	}
}

func TestADCTokenSource_TokenFetchError(t *testing.T) {
	// Create a fake service account JSON that FindDefaultCredentials can parse
	// but whose token endpoint is unreachable -- exercises the token fetch error path (line 34-35).
	fakeServiceAccount := `{
		"type": "service_account",
		"project_id": "fake-project",
		"private_key_id": "fake-key-id",
		"private_key": "-----BEGIN RSA PRIVATE KEY-----\nMIIEowIBAAKCAQEA2a2rwplBQLHgMH+kHNDwPMO86M08GdWRaSKhWJOxMM+xHPcb\nwsLhaa3dGNRCH9pp3VGwaIXBiwVX6XVJFnMF5gMFfbMM/Qv9bTJKEkMO3VOVB2q\nXQejBIU3f4b35CXQmkMSdPp7vPKzVDB0VJO2qG6eYfnJDLR2lD6pV2xWY6VaF0P\nvaxHPGuP0KRzs1g1sPmiDyBnMG1HzLVvn4W/dxVKFUfQ2MJsd+P8F/ov3bVHiibK\njNhJJw+4kFLJzDRIShSQ7SVEenNGS+rVuasZFxRGe6h4N2S5dJI4f2Xaaz0fJJE6\n2yCpwrAJVVsD5HB2TJQbNlBSH0oqBagkFcX4AwIDAQABAoIBAFEFTdPLJL0dAghE\nH4MX2bZh3KaxNxHVKTFgGCAA2G04bHJAOcWRjMQKX0SZ8t5NQ7A6qN8z6aSkPXnx\nvfZqPFbcI+pGyE7Rd25u1qDLdlJ3rIkIcXU4MkG6SpPPhWbwNnpJtx7r/x6MHZA6\nXFz4T6IXPUFXcQ/Y2hzgm3aVhQhgSKt06s3nLja1lZf7bhfOr4yF0uZ6sJYFP2j\nCVMo0NM7IJAM1NQMV0BjFhCjPbpKKvS4pqCj7FmfcGNJgzQSTfpLNJN5/W7TNtM\ndSjpabBgAEDnFaN1TGPJp6j5rX2RNSJ1FpK0rOfL83OQYX3/Kse3MfW8p/lCj/W3\n2n7VZ2ECgYEA8Zk6HDf3BH6X+mD/OUGX9HgQoJAVLBNrE3B6vVj/p3bNTCHPFUUJ\ncLMHj0t7u5bCxLPoqAQ/xNNsDmqI/cGRBMC4h3HBxiVJin+BTFVBSC6ETHO7MwIi\nvjBZ5W0UH3F2Nti7KCBaF7Gv2RTX6z8hD/8q5KMVuFx4EpGQCh17FuMCgYEA5n3S\nOiJFwrBRgPb4pRGzxMP/MfJWi0VxoW0jHCSNzaMLfRD1Wp4JqvO4kY0EFJXP9C0z\nSajgliVH9uy3YCLDl/bPlAVLjJM/4Fo3f7OdrUh4cCU7sCKq7Nd5NjFBW5IFblI6\nVf9T+VLH1UvGMXEPDL2NMYTbOwgvSCUM+pBGxEECgYEA0T/XEUqNkML1JIJy3KSq\n/zS3FT8BSRluaKr1zPajg6FlIxrjFhh/IWe2jibSKJ4BLRt2WqC9qJk32CaswpfF\nNY5RwNFe1Y4pRIkH6MFB8OY7fTm2EY4KaJDIRDoKqLPqXNjQLHXEpVhAj0PN5I+M\nKVZA3xrIbxdCBYKbBBJNflsCgYBmJCe+ZfLvTGjxqTv0hpuO3TjMr2r5ERZeKJ7n\nLOZ7PaZS9T+118gNZrUEwOkMXrrv0vBWCb5bRNBhr9eSzCm/e2hg5FFdpFb3GdVX\n5nXO7sPZkQrC1G1BhGjYnBT/8gFf8sMPOejh05uT8oVMpVpbSi3TEkRXe5SAgGl6\nMuGNAQKBgHfFrJMhKhHLz7AkRA58HGx/WAzGz2b7E+4q1uRibGCrq+IPxjasIUqJ\nhkHkPdnU4S0e6F6R/bGPulYKz+vy2pCDwb0kVS8V2pA9i0xnliZ/nPOSM7R49V+H\ng4MpPV4O3pB3KHFXsMnPDrLxcuLF2ipfNjS1TqTwxnRLICAGwEBR\n-----END RSA PRIVATE KEY-----\n",
		"client_email": "fake@fake-project.iam.gserviceaccount.com",
		"client_id": "123456789",
		"auth_uri": "https://accounts.google.com/o/oauth2/auth",
		"token_uri": "http://127.0.0.1:1/token",
		"auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
		"client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/fake"
	}`

	tmpDir := t.TempDir()
	credFile := filepath.Join(tmpDir, "creds.json")
	if err := os.WriteFile(credFile, []byte(fakeServiceAccount), 0600); err != nil {
		t.Fatal(err)
	}
	t.Setenv("GOOGLE_APPLICATION_CREDENTIALS", credFile)

	ts, err := ADCTokenSource(t.Context())
	if err != nil {
		t.Fatalf("ADCTokenSource should succeed with valid JSON: %v", err)
	}
	_, err = ts.Token(t.Context())
	// FindDefaultCredentials succeeds but token fetch fails (unreachable token_uri).
	if err == nil {
		t.Fatal("expected token fetch error")
	}
}

func TestADCTokenSource_TokenSuccess(t *testing.T) {
	// Set up a fake token server that returns a valid OAuth2 token response.
	tokenServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"access_token":"fake-access-token","token_type":"Bearer","expires_in":3600}`))
	}))
	defer tokenServer.Close()

	// Create a fake service account JSON pointing at the test token server.
	// Uses a real RSA key format (PKCS8) that google.FindDefaultCredentials can parse.
	fakeServiceAccount := `{
		"type": "service_account",
		"project_id": "fake-project",
		"private_key_id": "fake-key-id",
		"private_key": "-----BEGIN PRIVATE KEY-----\nMIIEvgIBADANBgkqhkiG9w0BAQEFAASCBKgwggSkAgEAAoIBAQCoCEnYL2HY07wl\nxKpMSI7+/AvoXwZZXHGyCnhxDMSuybys4KpE59CeHFvCD7Lkvtmz+iN04APvmUWH\nJEFEwMy877y1F1YiSkmI9dL0WNm31VBL8ar8mDSySr9x92KTfsQYDtXrmcA1colN\nwuhEFVWUfEzgbSOxVPH+izk0uE5jDwOfy+SRFdlW1maO1/KIcjHlGvGT5Ns1Sfik\ndq0MOqFWPbcVGeeUwPUgJWo5R6VNIL0iZfO8HX8qMMpSENbkAc7fh+VY2+ZbaYwt\nDvWFtg95+3LoPPJkAR/ZMTyz7DjBh2YHlEB23BC/SKGw+84jZJVLehDpGkNGk1R6\nUk/NofBdAgMBAAECggEAAmxuLjtOuLsjE1NhFm0UfOqDPHSEaa0K6cO7ZXwG6TN5\nMHAkHI7QZDtp+mRSEvkKuE0TDlg9zkqiZVbZRyXVviLiEKWXxTJXma1b/9C5QSzH\nNfU0H2UNm4W2+ayVBCYXq3YpUTbcPhkbEF9PjM3W+GeLj1Abq+7a964n4tIGtWiH\nw032xepBfwH/s9/t63lY8RaHBruIjbaon0XVZRKNskdzCfW0Pq25xSdiBvUDYrMz\nifnoi69oHqROxNycdY9IuPMDuc3rmXxKirN7UPm+wpvys9o2ZcLi88B9E4kM2a8x\nBc/o5ZUsmy+Ze/CGb57z5+D6pPeD7QCpiYqw6UTdGQKBgQDmrR/42DKo7r5ZGj93\nlvtyOR+oT/JtNjOqYnJKZSeQUUhdsDRalc2u2VhLLvu2IFAU8aO3Xjvs0efc9ouo\nmQt3TXlOrxTcN0XH8lS/+xs74rXTnf96HFVNkgZLCCQPveNpTxx3i4EGjIPA5EM2\nwfsaWdb27WbKi98kcDlg0MXFxQKBgQC6eqImY6L1kMmF8LwDq3NAMnS4tVcITSnZ\njLW7cSBDpE0UOOOZfFn/9jBDTojHGj1luMvYIpUu8BqBHaslLH73S7aFpmuX8e5F\nR+yCvSpNKhJrRwNglvJhx30NJ1AUO3zL5yLkNRG60OB03T0x+9BjOXALTjFM2bCW\noMBRYHNBuQKBgQDbCjPc4LaiMSnwg/sWPPkBCnskIN4rlBdVSGwMdqct4/EafZIJ\nHkyUJnAv3CpKU76XVKjIGyQ+CUfpSvdsnf8ERz3UWG4vehC5/0M4lWHT6MANdO89\n7Z+Wq/1rzAwqIn7J5cQ7Q+294TnOtTGZ0nL1H6//A4ji9hRqjmH9q+DQnQKBgFpL\nCPqXog9PhRSyxQbt3IdIJxZM0BB39HyfUlupYhr+kkfpZ+MowBddKG9etoHZpcL8\nYM+NpzisD9lW+Uitq+ioI3/BXjWbcmjfc5i2aaYlaffB0dPSIxjPVDCrSW4Sg9Hj\nYBSp0aTogNZ1Ta1HJdb1t9fxi5OPkJ7OxXBhyE0ZAoGBALPSMchZpi6ufx7Tan1H\nMlOHLkEgyb1ZG+haVk+iYLfd7WVnep4kkmFcCQAVr/QnVvpeducDQFtsh6VCVgCP\nVMGHa5YODt6YsoD/s2kq3GGfnJ/8UEf0Rw8wYOPBbBXra8hysuNRMILyrg4NgUHY\n88Ch7NOHvR0rRdbNpppNvJqs\n-----END PRIVATE KEY-----\n",
		"client_email": "fake@fake-project.iam.gserviceaccount.com",
		"client_id": "123456789",
		"auth_uri": "https://accounts.google.com/o/oauth2/auth",
		"token_uri": "TOKEN_URI_PLACEHOLDER",
		"auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
		"client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/fake"
	}`

	// Replace placeholder with actual test server URL.
	fakeServiceAccount = strings.Replace(fakeServiceAccount, "TOKEN_URI_PLACEHOLDER", tokenServer.URL, 1)

	tmpDir := t.TempDir()
	credFile := filepath.Join(tmpDir, "creds.json")
	if err := os.WriteFile(credFile, []byte(fakeServiceAccount), 0600); err != nil {
		t.Fatal(err)
	}
	t.Setenv("GOOGLE_APPLICATION_CREDENTIALS", credFile)

	ts, err := ADCTokenSource(t.Context())
	if err != nil {
		t.Fatalf("ADCTokenSource failed: %v", err)
	}
	tok, err := ts.Token(t.Context())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if tok != "fake-access-token" {
		t.Errorf("token = %q, want fake-access-token", tok)
	}
}

// --- BF.9 Item 2: Vercel env var alias tests ---

func TestVercelEnvVar_Project(t *testing.T) {
	t.Setenv("GOOGLE_VERTEX_PROJECT", "vercel-proj")
	t.Setenv("GOOGLE_CLOUD_PROJECT", "")
	t.Setenv("GCLOUD_PROJECT", "")

	model := Chat("m", WithTokenSource(provider.StaticToken("tok")))
	cm := model.(*chatModel)
	if cm.opts.project != "vercel-proj" {
		t.Errorf("project = %q, want vercel-proj", cm.opts.project)
	}
}

func TestVercelEnvVar_Location(t *testing.T) {
	t.Setenv("GOOGLE_VERTEX_LOCATION", "europe-west1")
	t.Setenv("GOOGLE_CLOUD_LOCATION", "")

	model := Chat("m", WithTokenSource(provider.StaticToken("tok")))
	cm := model.(*chatModel)
	if cm.opts.location != "europe-west1" {
		t.Errorf("location = %q, want europe-west1", cm.opts.location)
	}
}

func TestEnvVarPriority_ProjectExplicitFirst(t *testing.T) {
	t.Setenv("GOOGLE_VERTEX_PROJECT", "vercel-proj")
	t.Setenv("GOOGLE_CLOUD_PROJECT", "cloud-proj")

	// Explicit WithProject should win over all env vars.
	model := Chat("m", WithTokenSource(provider.StaticToken("tok")), WithProject("explicit-proj"))
	cm := model.(*chatModel)
	if cm.opts.project != "explicit-proj" {
		t.Errorf("project = %q, want explicit-proj", cm.opts.project)
	}
}

func TestEnvVarPriority_VertexBeforeCloud(t *testing.T) {
	t.Setenv("GOOGLE_VERTEX_PROJECT", "vercel-proj")
	t.Setenv("GOOGLE_CLOUD_PROJECT", "cloud-proj")
	t.Setenv("GCLOUD_PROJECT", "gcloud-proj")

	model := Chat("m", WithTokenSource(provider.StaticToken("tok")))
	cm := model.(*chatModel)
	if cm.opts.project != "vercel-proj" {
		t.Errorf("project = %q, want vercel-proj (GOOGLE_VERTEX_PROJECT takes priority)", cm.opts.project)
	}
}

func TestEnvVarPriority_LocationExplicitFirst(t *testing.T) {
	t.Setenv("GOOGLE_VERTEX_LOCATION", "vercel-loc")
	t.Setenv("GOOGLE_CLOUD_LOCATION", "cloud-loc")

	model := Chat("m", WithTokenSource(provider.StaticToken("tok")), WithLocation("explicit-loc"))
	cm := model.(*chatModel)
	if cm.opts.location != "explicit-loc" {
		t.Errorf("location = %q, want explicit-loc", cm.opts.location)
	}
}

func TestEnvVarPriority_VertexLocationBeforeCloud(t *testing.T) {
	t.Setenv("GOOGLE_VERTEX_LOCATION", "vercel-loc")
	t.Setenv("GOOGLE_CLOUD_LOCATION", "cloud-loc")

	model := Chat("m", WithTokenSource(provider.StaticToken("tok")))
	cm := model.(*chatModel)
	if cm.opts.location != "vercel-loc" {
		t.Errorf("location = %q, want vercel-loc (GOOGLE_VERTEX_LOCATION takes priority)", cm.opts.location)
	}
}

func TestEnvVarFallback_CloudProject(t *testing.T) {
	t.Setenv("GOOGLE_VERTEX_PROJECT", "")
	t.Setenv("GOOGLE_CLOUD_PROJECT", "cloud-proj")
	t.Setenv("GCLOUD_PROJECT", "gcloud-proj")

	model := Chat("m", WithTokenSource(provider.StaticToken("tok")))
	cm := model.(*chatModel)
	if cm.opts.project != "cloud-proj" {
		t.Errorf("project = %q, want cloud-proj", cm.opts.project)
	}
}

func TestEnvVarFallback_GcloudProject(t *testing.T) {
	t.Setenv("GOOGLE_VERTEX_PROJECT", "")
	t.Setenv("GOOGLE_CLOUD_PROJECT", "")
	t.Setenv("GCLOUD_PROJECT", "gcloud-proj")

	model := Chat("m", WithTokenSource(provider.StaticToken("tok")))
	cm := model.(*chatModel)
	if cm.opts.project != "gcloud-proj" {
		t.Errorf("project = %q, want gcloud-proj", cm.opts.project)
	}
}

// --- BF.9 Item 3: URL path tests ---

func TestChatURL_UsesOpenAPI(t *testing.T) {
	transport := &urlCapturingTransport{}
	model := Chat("gemini-2.5-pro",
		WithTokenSource(provider.StaticToken("tok")),
		WithProject("my-project"),
		WithLocation("us-central1"),
		WithHTTPClient(&http.Client{Transport: transport}),
	)
	_, _ = model.(*chatModel).DoGenerate(t.Context(), provider.GenerateParams{
		Messages: []provider.Message{
			{Role: provider.RoleUser, Content: []provider.Part{{Type: provider.PartText, Text: "hi"}}},
		},
	})
	// Chat uses /endpoints/openapi (OpenAI-compat)
	if !strings.Contains(transport.captured, "/endpoints/openapi/chat/completions") {
		t.Errorf("chat URL should use /endpoints/openapi, got %q", transport.captured)
	}
}

func TestEmbeddingURL_UsesPublishersGoogle(t *testing.T) {
	transport := &urlCapturingTransport{}
	model := Embedding("text-embedding-004",
		WithTokenSource(provider.StaticToken("tok")),
		WithProject("my-project"),
		WithLocation("us-central1"),
		WithHTTPClient(&http.Client{Transport: transport}),
	)
	_, _ = model.DoEmbed(t.Context(), []string{"hello"}, provider.EmbedParams{})
	// Embedding uses /publishers/google (native Vertex API)
	if !strings.Contains(transport.captured, "/publishers/google/models/text-embedding-004:predict") {
		t.Errorf("embedding URL should use /publishers/google, got %q", transport.captured)
	}
}

func TestImageURL_UsesPublishersGoogle(t *testing.T) {
	transport := &urlCapturingTransport{}
	model := Image("imagen-3.0-generate-002",
		WithTokenSource(provider.StaticToken("tok")),
		WithProject("my-project"),
		WithLocation("us-central1"),
		WithHTTPClient(&http.Client{Transport: transport}),
	)
	_, _ = model.DoGenerate(t.Context(), provider.ImageParams{
		Prompt: "a cat",
		N:      1,
	})
	// Image uses /publishers/google (native Vertex API)
	if !strings.Contains(transport.captured, "/publishers/google/models/imagen-3.0-generate-002:predict") {
		t.Errorf("image URL should use /publishers/google, got %q", transport.captured)
	}
}

// --- BF.9: Env var aliases work for Embedding/Image too ---

func TestEmbedding_VercelEnvVars(t *testing.T) {
	t.Setenv("GOOGLE_VERTEX_PROJECT", "vercel-proj")
	t.Setenv("GOOGLE_VERTEX_LOCATION", "vercel-loc")
	t.Setenv("GOOGLE_CLOUD_PROJECT", "")
	t.Setenv("GOOGLE_CLOUD_LOCATION", "")
	t.Setenv("GCLOUD_PROJECT", "")

	transport := &urlCapturingTransport{}
	model := Embedding("text-embedding-004",
		WithTokenSource(provider.StaticToken("tok")),
		WithHTTPClient(&http.Client{Transport: transport}),
	)
	_, _ = model.DoEmbed(t.Context(), []string{"hello"}, provider.EmbedParams{})

	expected := "https://vercel-loc-aiplatform.googleapis.com/v1beta1/projects/vercel-proj/locations/vercel-loc/publishers/google/models/text-embedding-004:predict"
	if transport.captured != expected {
		t.Errorf("URL = %q, want %q", transport.captured, expected)
	}
}

func TestImage_VercelEnvVars(t *testing.T) {
	t.Setenv("GOOGLE_VERTEX_PROJECT", "vercel-proj")
	t.Setenv("GOOGLE_VERTEX_LOCATION", "vercel-loc")
	t.Setenv("GOOGLE_CLOUD_PROJECT", "")
	t.Setenv("GOOGLE_CLOUD_LOCATION", "")
	t.Setenv("GCLOUD_PROJECT", "")

	transport := &urlCapturingTransport{}
	model := Image("imagen-3.0-generate-002",
		WithTokenSource(provider.StaticToken("tok")),
		WithHTTPClient(&http.Client{Transport: transport}),
	)
	_, _ = model.DoGenerate(t.Context(), provider.ImageParams{Prompt: "cat", N: 1})

	expected := "https://vercel-loc-aiplatform.googleapis.com/v1beta1/projects/vercel-proj/locations/vercel-loc/publishers/google/models/imagen-3.0-generate-002:predict"
	if transport.captured != expected {
		t.Errorf("URL = %q, want %q", transport.captured, expected)
	}
}

// Note: urlCapturingTransport is defined in vertex_test.go and shared
// across all test files in the same package.
