# OAuth Integration Guide for MemoryGate

## New Files Created

1. `oauth_models.py` - User, session, and API key models
2. `oauth.py` - OAuth provider integrations (Google, GitHub)
3. `auth_middleware.py` - Authentication middleware and utilities
4. `oauth_routes.py` - OAuth flow endpoints
5. `.env.example` - OAuth configuration template
6. `requirements.txt` - Updated with bcrypt dependency

## Integration Steps

### 1. Update `server.py` imports

Add these imports at the top of `server.py`:

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from oauth_models import User, UserSession, APIKey, OAuthState
from oauth_routes import router as auth_router
```

### 2. Initialize OAuth tables in `init_db()`

After `Base.metadata.create_all(engine)`, ensure OAuth models are imported:

```python
# In init_db() function, after existing table creation:
# Import oauth_models to register tables with Base
import oauth_models
Base.metadata.create_all(engine)
```

### 3. Create FastAPI app and mount OAuth routes

Replace or modify the MCP server initialization to include FastAPI:

```python
# Create lifespan context manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle startup and shutdown"""
    init_db()
    init_http_client()
    yield
    cleanup_http_client()

# Create FastAPI app
app = FastAPI(title="MemoryGate", lifespan=lifespan)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[os.environ.get("FRONTEND_URL", "http://localhost:3000")],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount OAuth routes
app.include_router(auth_router)

# Create MCP server and mount to /mcp
mcp = FastMCP("MemoryGate")
app.mount("/mcp", mcp.app)
```

### 4. Update database dependency in oauth_routes.py

Replace the placeholder `get_db_session()` in `oauth_routes.py`:

```python
# At the top of oauth_routes.py, replace:
def get_db_session():
    """Database dependency"""
    from server import SessionLocal
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
```

### 5. Add environment variables

Copy `.env.example` to `.env` and fill in OAuth credentials:

```bash
cp .env.example .env
# Edit .env with your OAuth credentials
```

### 6. Install new dependencies

```bash
pip install -r requirements.txt
```

### 7. Run database migrations

The new tables will be created automatically on next startup due to `Base.metadata.create_all()`,
but for production, consider using Alembic migrations:

```bash
# Generate migration
alembic revision --autogenerate -m "Add OAuth support"

# Apply migration
alembic upgrade head
```

## OAuth Flow

1. User visits `/auth/login/google` or `/auth/login/github`
2. Server generates PKCE challenge and OAuth state
3. User redirects to provider (Google/GitHub)
4. Provider redirects back to `/auth/callback/{provider}`
5. Server exchanges code for token, fetches user info
6. Server creates/updates user, creates session
7. Server sets `mg_session` cookie and redirects to frontend

## API Authentication

### Session Authentication (Browser)
- Cookie: `mg_session=<token>`
- Managed automatically by OAuth flow
- 7-day expiry, auto-refresh on activity

### API Key Authentication (MCP Tools)
- Header: `Authorization: Bearer mg_<key>`
- OR Header: `X-API-Key: mg_<key>`
- Create via `POST /auth/api-keys`
- Manage via `/auth/api-keys` endpoints

## Endpoints

### OAuth Flow
- `GET /auth/login/{provider}` - Start OAuth flow
- `GET /auth/callback/{provider}` - OAuth callback
- `POST /auth/logout` - Logout and revoke session

### User Management
- `GET /auth/me` - Get current user info

### API Keys
- `POST /auth/api-keys` - Create new API key
- `GET /auth/api-keys` - List user's API keys
- `DELETE /auth/api-keys/{key_id}` - Revoke API key

## Adding Authentication to MCP Tools

To protect MCP tools with authentication, use the `require_auth` dependency:

```python
from auth_middleware import require_auth, get_current_user
from oauth_models import User

@mcp.tool()
async def protected_tool(user: User = Depends(require_auth)):
    """This tool requires authentication"""
    # user is automatically populated
    return f"Hello {user.email}!"

@mcp.tool()
async def optional_auth_tool(user: User = Depends(get_current_user)):
    """This tool works with or without authentication"""
    if user:
        return f"Hello {user.email}!"
    else:
        return "Hello anonymous user!"
```

## Production Deployment Checklist

- [ ] Set `OAUTH_REDIRECT_BASE=https://memorygate.fly.dev`
- [ ] Set `FRONTEND_URL=https://memorygate.ai`
- [ ] Configure OAuth redirect URIs in Google/GitHub:
  - Google: `https://memorygate.fly.dev/auth/callback/google`
  - GitHub: `https://memorygate.fly.dev/auth/callback/github`
- [ ] Enable secure cookies (HTTPS only)
- [ ] Set up CORS for production frontend domain
- [ ] Configure session cleanup cron job
- [ ] Monitor API key usage and rate limits

## Security Notes

- API keys are hashed with bcrypt (never stored in plain text)
- Session tokens are secure random (48 bytes urlsafe)
- OAuth state uses PKCE for additional security
- All sessions have 7-day expiry with activity refresh
- HTTPS required in production (cookies marked `secure=True`)
- OAuth state expires after 10 minutes
- Failed auth attempts should be rate-limited (add middleware)
