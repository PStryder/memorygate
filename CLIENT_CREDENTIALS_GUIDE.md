# MemoryGate Client Credentials - Quick Start

## For PStryder's Desktop → MemoryGate Authentication

### Step 1: Set Server Secrets

```bash
fly secrets set PSTRYDER_DESKTOP=your-client-id
fly secrets set PSTRYDER_DESKTOP_SECRET=your-client-secret
```

### Step 2: Exchange Credentials for API Key

**One-time setup** - Call this once to get your API key:

```bash
curl -X POST https://memorygate.fly.dev/auth/client \
  -H "Content-Type: application/json" \
  -d '{
    "client_id": "your-client-id",
    "client_secret": "your-client-secret"
  }'
```

**Response:**
```json
{
  "api_key": "mg_wXyZ1234567890abcdefghijklmnopqrstuv",
  "key_prefix": "mg_wXyZ1234",
  "user_id": "uuid-here",
  "expires_at": null
}
```

**IMPORTANT:** Save the `api_key` value immediately! It's only shown once.

### Step 3: Store API Key Locally

```bash
# Add to your .env or environment
export MEMORYGATE_API_KEY=mg_wXyZ1234567890abcdefghijklmnopqrstuv
```

### Step 4: Configure MCP Client

The MemoryGate MCP server **requires authentication**. All `/mcp/*` endpoints are protected by API key.

**Example MCP configuration:**

```json
{
  "memorygate": {
    "url": "https://memorygate.fly.dev/mcp/",
    "headers": {
      "Authorization": "Bearer ${MEMORYGATE_API_KEY}"
    }
  }
}
```

Or use the `X-API-Key` header:

```json
{
  "memorygate": {
    "url": "https://memorygate.fly.dev/mcp/",
    "headers": {
      "X-API-Key": "${MEMORYGATE_API_KEY}"
    }
  }
}
```

**IMPORTANT:** Without a valid API key, all MCP requests will return `401 Unauthorized`.

### Testing Authentication

Test that your API key works:

```bash
curl https://memorygate.fly.dev/auth/me \
  -H "Authorization: Bearer mg_wXyZ1234567890abcdefghijklmnopqrstuv"
```

Expected response:
```json
{
  "id": "uuid",
  "email": "PSTRYDER_DESKTOP@client.memorygate.internal",
  "name": "PSTRYDER_DESKTOP",
  "oauth_provider": "client_credentials",
  "is_verified": true,
  "created_at": "2026-01-03T..."
}
```

### Managing Your API Key

**List your API keys:**
```bash
curl https://memorygate.fly.dev/auth/api-keys \
  -H "Authorization: Bearer mg_wXyZ..."
```

**Revoke a compromised key:**
```bash
curl -X DELETE https://memorygate.fly.dev/auth/api-keys/{key_id} \
  -H "Authorization: Bearer mg_wXyZ..."
```

**Generate a new key** (if original is lost):
Just call `/auth/client` again - it will create a new API key for the same user.

### Security Notes

- Client credentials (`PSTRYDER_DESKTOP` + secret) are stored server-side only
- API key never expires (manual revocation if compromised)
- API key hashed with bcrypt in database (never plaintext)
- User created as: `PSTRYDER_DESKTOP@client.memorygate.internal`
- Provider type: `client_credentials` (distinct from OAuth users)

### Troubleshooting

**401 Unauthorized:**
- Check client_id matches `PSTRYDER_DESKTOP` exactly
- Check client_secret matches server secret exactly
- Verify secrets are set on Fly.io: `fly secrets list`

**503 Service Unavailable:**
- Server secrets not configured
- Run: `fly secrets set PSTRYDER_DESKTOP=... PSTRYDER_DESKTOP_SECRET=...`

**API key not working:**
- Verify using correct header: `Authorization: Bearer mg_...` or `X-API-Key: mg_...`
- Check key not revoked: `GET /auth/api-keys`
- Regenerate if lost: `POST /auth/client` with credentials
