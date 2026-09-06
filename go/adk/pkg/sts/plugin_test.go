package sts

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"log/slog"

	"github.com/golang-jwt/jwt/v5"
	kagentmodels "github.com/kagent-dev/kagent/go/adk/pkg/models"
	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/session"
	"google.golang.org/genai"
)

type fakeInvocationContext struct {
	context.Context
	sessionID string
	ended     bool
}

func (f fakeInvocationContext) Agent() agent.Agent              { return nil }
func (f fakeInvocationContext) Artifacts() agent.Artifacts      { return nil }
func (f fakeInvocationContext) Memory() agent.Memory            { return nil }
func (f fakeInvocationContext) Session() session.Session        { return fakeSession{id: f.sessionID} }
func (f fakeInvocationContext) InvocationID() string            { return "" }
func (f fakeInvocationContext) Branch() string                  { return "" }
func (f fakeInvocationContext) IsolationScope() string          { return "" }
func (f fakeInvocationContext) UserContent() *genai.Content     { return nil }
func (f fakeInvocationContext) RunConfig() *agent.RunConfig     { return nil }
func (f *fakeInvocationContext) EndInvocation()                 { f.ended = true }
func (f fakeInvocationContext) Ended() bool                     { return f.ended }
func (f fakeInvocationContext) ResumedInput(string) (any, bool) { return nil, false }
func (f fakeInvocationContext) WithContext(ctx context.Context) agent.InvocationContext {
	f.Context = ctx
	return &f
}
func (f fakeInvocationContext) WithICDelta(*agent.InvocationContextDelta) agent.InvocationContext {
	return &f
}

type fakeSession struct {
	id string
}

func (f fakeSession) ID() string                { return f.id }
func (f fakeSession) AppName() string           { return "" }
func (f fakeSession) UserID() string            { return "" }
func (f fakeSession) State() session.State      { return nil }
func (f fakeSession) Events() session.Events    { return nil }
func (f fakeSession) LastUpdateTime() time.Time { return time.Time{} }

// TestHeaderProvider_SurvivesContextWrapping is a regression test for
// solo-io/kagent-enterprise#2490: the MCP client library wraps the context via
// context.WithValue before HeaderProvider sees it, so any concrete type
// (including a SessionID() method) is no longer type-assertable -- only
// ctx.Value lookups for keys already set on an ancestor context survive. This
// wraps the context the same way to prove the fix (looking up the bearer token
// via ctx.Value, not a type-asserted session ID) works under that condition.
func TestHeaderProvider_SurvivesContextWrapping(t *testing.T) {
	t.Parallel()
	plugin := NewTokenPropagationPlugin(nil, slog.New(slog.DiscardHandler), nil, nil)
	plugin.setCachedToken(tokenCacheKey("subject-token"), "token-abc", 0)

	ctx := context.WithValue(context.Background(), kagentmodels.BearerTokenKey, "subject-token")
	// Simulate the MCP client library wrapping the context with an unrelated key,
	// as it does in practice -- this is what defeats a type-assertion-based lookup.
	type unrelatedKey struct{}
	ctx = context.WithValue(ctx, unrelatedKey{}, "irrelevant")

	headers := plugin.HeaderProvider(ctx)

	if headers["Authorization"] != "Bearer token-abc" {
		t.Fatalf("Authorization header = %q, want %q", headers["Authorization"], "Bearer token-abc")
	}
}

// TestHeaderProvider_NoBearerTokenInContext confirms HeaderProvider degrades
// gracefully (falls back to existing headers) when there's nothing to look up,
// rather than erroring.
func TestHeaderProvider_NoBearerTokenInContext(t *testing.T) {
	t.Parallel()
	plugin := NewTokenPropagationPlugin(nil, slog.New(slog.DiscardHandler), nil, nil)
	plugin.setCachedToken(tokenCacheKey("subject-token"), "token-abc", 0)

	headers := plugin.HeaderProvider(context.Background())

	if headers != nil {
		t.Fatalf("headers = %v, want nil", headers)
	}
}

func TestBeforeRunCallback_ReusesCachedDynamicActorTokenForExchange(t *testing.T) {
	t.Parallel()

	fetchCount := 0
	exchangeCount := 0
	var srv *httptest.Server
	srv = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/.well-known/oauth-authorization-server" {
			_ = json.NewEncoder(w).Encode(map[string]any{
				"issuer":         srv.URL,
				"token_endpoint": srv.URL + "/token",
			})
			return
		}
		if r.URL.Path != "/token" {
			http.NotFound(w, r)
			return
		}
		exchangeCount++
		if err := r.ParseForm(); err != nil {
			t.Fatalf("ParseForm() error = %v", err)
		}
		if got := r.FormValue("actor_token"); got != "dynamic-actor" {
			t.Fatalf("actor_token = %q, want %q", got, "dynamic-actor")
		}
		_ = json.NewEncoder(w).Encode(map[string]any{
			"access_token":      "access-token",
			"issued_token_type": string(TokenTypeJWT),
		})
	}))
	defer srv.Close()

	integration, err := NewSTSIntegration(
		srv.URL+"/.well-known/oauth-authorization-server",
		"",
		func(context.Context) (string, error) {
			fetchCount++
			return "dynamic-actor", nil
		},
		nil,
		5,
		true,
		false,
	)
	if err != nil {
		t.Fatalf("NewSTSIntegration() error = %v", err)
	}

	plugin := NewTokenPropagationPlugin(integration, slog.New(slog.DiscardHandler), nil, nil)
	// Two different callers (distinct subject tokens, the cache key since #2490) in
	// the same run: each gets its own STS exchange, but the dynamically-fetched
	// actor token is cached and reused across both, since actor-token caching is
	// independent of caller identity.
	for _, subjectToken := range []string{"subject-token-one", "subject-token-two"} {
		ctx := context.WithValue(context.Background(), kagentmodels.BearerTokenKey, subjectToken)
		if _, err := plugin.BeforeRunCallback(&fakeInvocationContext{
			Context:   ctx,
			sessionID: "sess-shared",
		}); err != nil {
			t.Fatalf("BeforeRunCallback() error = %v", err)
		}
	}

	if fetchCount != 1 {
		t.Fatalf("fetchActorToken calls = %d, want 1", fetchCount)
	}
	if exchangeCount != 2 {
		t.Fatalf("token exchange calls = %d, want 2", exchangeCount)
	}
}

// TestBeforeRunCallback_SameBearerTokenAcrossSessionsSharesExchange is a
// regression test for kagent-dev/kagent#2181: keying the cache by bearer token
// instead of session ID means two different sessions presenting the *same*
// caller's token correctly share one exchange, and -- the actual point of
// #2181 -- two *different* callers sharing the *same* session ID (e.g. a
// shared A2A conversation) each still get their own, since the key follows the
// caller, not the conversation.
func TestBeforeRunCallback_SameBearerTokenAcrossSessionsSharesExchange(t *testing.T) {
	t.Parallel()

	exchangeCount := 0
	var srv *httptest.Server
	srv = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/.well-known/oauth-authorization-server" {
			_ = json.NewEncoder(w).Encode(map[string]any{
				"issuer":         srv.URL,
				"token_endpoint": srv.URL + "/token",
			})
			return
		}
		if r.URL.Path != "/token" {
			http.NotFound(w, r)
			return
		}
		exchangeCount++
		_ = json.NewEncoder(w).Encode(map[string]any{
			"access_token":      "access-token",
			"issued_token_type": string(TokenTypeJWT),
		})
	}))
	defer srv.Close()

	integration, err := NewSTSIntegration(
		srv.URL+"/.well-known/oauth-authorization-server",
		"", nil, nil, 5, true, false,
	)
	if err != nil {
		t.Fatalf("NewSTSIntegration() error = %v", err)
	}

	plugin := NewTokenPropagationPlugin(integration, slog.New(slog.DiscardHandler), nil, nil)

	// Same caller, two different sessions: one exchange, shared.
	for _, sessionID := range []string{"sess-one", "sess-two"} {
		ctx := context.WithValue(context.Background(), kagentmodels.BearerTokenKey, "same-caller-token")
		if _, err := plugin.BeforeRunCallback(&fakeInvocationContext{Context: ctx, sessionID: sessionID}); err != nil {
			t.Fatalf("BeforeRunCallback() error = %v", err)
		}
	}
	if exchangeCount != 1 {
		t.Fatalf("token exchange calls after same-caller/different-sessions = %d, want 1", exchangeCount)
	}

	// Two different callers sharing one session ID (the #2181 scenario): each
	// still gets its own exchange, since identity follows the token, not the session.
	for _, subjectToken := range []string{"caller-a-token", "caller-b-token"} {
		ctx := context.WithValue(context.Background(), kagentmodels.BearerTokenKey, subjectToken)
		if _, err := plugin.BeforeRunCallback(&fakeInvocationContext{Context: ctx, sessionID: "shared-session"}); err != nil {
			t.Fatalf("BeforeRunCallback() error = %v", err)
		}
	}
	if exchangeCount != 3 {
		t.Fatalf("token exchange calls after two different callers sharing a session = %d, want 3 (1 shared + 2 distinct)", exchangeCount)
	}
}

func TestBeforeRunCallback_SendsResourceAndAudience(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name         string
		resource     []string
		audience     []string
		wantResource string
		wantAudience string
	}{
		{
			name:         "configured target is sent",
			resource:     []string{"https://mcp.example.com"},
			audience:     []string{"mcp-backend"},
			wantResource: "https://mcp.example.com",
			wantAudience: "mcp-backend",
		},
		{
			name:         "no target leaves resource and audience unset",
			wantResource: "",
			wantAudience: "",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			type exchangeForm struct {
				resource string
				audience string
				err      error
			}
			// Buffered so the handler never blocks on send; the value is read
			// back on the test goroutine to avoid a data race on the captured form.
			gotForm := make(chan exchangeForm, 1)

			var srv *httptest.Server
			srv = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if r.URL.Path == "/.well-known/oauth-authorization-server" {
					_ = json.NewEncoder(w).Encode(map[string]any{
						"issuer":         srv.URL,
						"token_endpoint": srv.URL + "/token",
					})
					return
				}
				if r.URL.Path != "/token" {
					http.NotFound(w, r)
					return
				}
				if err := r.ParseForm(); err != nil {
					gotForm <- exchangeForm{err: err}
				} else {
					gotForm <- exchangeForm{resource: r.FormValue("resource"), audience: r.FormValue("audience")}
				}
				_ = json.NewEncoder(w).Encode(map[string]any{
					"access_token":      "access-token",
					"issued_token_type": string(TokenTypeJWT),
				})
			}))
			defer srv.Close()

			integration, err := NewSTSIntegration(
				srv.URL+"/.well-known/oauth-authorization-server",
				"", nil, nil, 5, true, false,
			)
			if err != nil {
				t.Fatalf("NewSTSIntegration() error = %v", err)
			}

			plugin := NewTokenPropagationPlugin(integration, slog.New(slog.DiscardHandler), tt.resource, tt.audience)
			ctx := context.WithValue(context.Background(), kagentmodels.BearerTokenKey, "subject-token")
			if _, err := plugin.BeforeRunCallback(&fakeInvocationContext{
				Context:   ctx,
				sessionID: "sess-resource",
			}); err != nil {
				t.Fatalf("BeforeRunCallback() error = %v", err)
			}

			select {
			case got := <-gotForm:
				if got.err != nil {
					t.Fatalf("ParseForm() error = %v", got.err)
				}
				if got.resource != tt.wantResource {
					t.Fatalf("resource = %q, want %q", got.resource, tt.wantResource)
				}
				if got.audience != tt.wantAudience {
					t.Fatalf("audience = %q, want %q", got.audience, tt.wantAudience)
				}
			case <-time.After(5 * time.Second):
				t.Fatal("timed out waiting for token exchange request")
			}
		})
	}
}

func TestExtractJWTExpiryUsesUnverifiedClaims(t *testing.T) {
	t.Parallel()
	want := time.Now().Add(time.Hour).Unix()
	token, err := jwt.NewWithClaims(jwt.SigningMethodHS256, jwt.MapClaims{
		"exp": want,
	}).SignedString([]byte("secret"))
	if err != nil {
		t.Fatalf("failed to sign token: %v", err)
	}

	if got := extractJWTExpiry(token); got != want {
		t.Fatalf("extractJWTExpiry() = %d, want %d", got, want)
	}
}
