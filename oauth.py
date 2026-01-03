"""
OAuth 2.0 Provider Integration for MemoryGate

Supports multiple OAuth providers (Google, GitHub) with unified interface
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Dict, Any
import httpx
import secrets
import hashlib
import base64
from urllib.parse import urlencode


@dataclass
class OAuthConfig:
    """Configuration for an OAuth provider"""
    client_id: str
    client_secret: str
    authorize_url: str
    token_url: str
    userinfo_url: str
    scopes: list[str]
    provider_name: str


@dataclass
class OAuthUserInfo:
    """Normalized user information from OAuth provider"""
    subject: str  # Provider's user ID
    email: str
    name: Optional[str]
    avatar_url: Optional[str]
    email_verified: bool
    raw_data: Dict[str, Any]  # Original provider response



class OAuthProvider(ABC):
    """Base class for OAuth providers"""
    
    def __init__(self, config: OAuthConfig):
        self.config = config
        self.http_client = httpx.Client(timeout=10.0)
    
    def generate_pkce_pair(self) -> tuple[str, str]:
        """Generate PKCE code_verifier and code_challenge"""
        code_verifier = secrets.token_urlsafe(64)
        code_challenge = base64.urlsafe_b64encode(
            hashlib.sha256(code_verifier.encode()).digest()
        ).decode().rstrip('=')
        return code_verifier, code_challenge
    
    def get_authorization_url(self, state: str, redirect_uri: str, 
                            code_challenge: Optional[str] = None) -> str:
        """Generate OAuth authorization URL"""
        params = {
            'client_id': self.config.client_id,
            'redirect_uri': redirect_uri,
            'response_type': 'code',
            'scope': ' '.join(self.config.scopes),
            'state': state,
        }
        
        if code_challenge:
            params['code_challenge'] = code_challenge
            params['code_challenge_method'] = 'S256'
        
        return f"{self.config.authorize_url}?{urlencode(params)}"
    
    def exchange_code(self, code: str, redirect_uri: str, 
                     code_verifier: Optional[str] = None) -> Dict[str, Any]:
        """Exchange authorization code for access token"""
        data = {
            'client_id': self.config.client_id,
            'client_secret': self.config.client_secret,
            'code': code,
            'redirect_uri': redirect_uri,
            'grant_type': 'authorization_code',
        }
        
        if code_verifier:
            data['code_verifier'] = code_verifier
        
        response = self.http_client.post(
            self.config.token_url,
            data=data,
            headers={'Accept': 'application/json'}
        )
        response.raise_for_status()
        return response.json()
    
    @abstractmethod
    def get_user_info(self, access_token: str) -> OAuthUserInfo:
        """Fetch user information using access token"""
        pass
    
    def __del__(self):
        """Clean up HTTP client"""
        if hasattr(self, 'http_client'):
            self.http_client.close()


class GoogleOAuth(OAuthProvider):
    """Google OAuth 2.0 provider"""
    
    def __init__(self, client_id: str, client_secret: str):
        config = OAuthConfig(
            client_id=client_id,
            client_secret=client_secret,
            authorize_url='https://accounts.google.com/o/oauth2/v2/auth',
            token_url='https://oauth2.googleapis.com/token',
            userinfo_url='https://www.googleapis.com/oauth2/v2/userinfo',
            scopes=['openid', 'email', 'profile'],
            provider_name='google'
        )
        super().__init__(config)
    
    def get_user_info(self, access_token: str) -> OAuthUserInfo:
        """Fetch Google user information"""
        response = self.http_client.get(
            self.config.userinfo_url,
            headers={'Authorization': f'Bearer {access_token}'}
        )
        response.raise_for_status()
        data = response.json()
        
        return OAuthUserInfo(
            subject=data['id'],
            email=data['email'],
            name=data.get('name'),
            avatar_url=data.get('picture'),
            email_verified=data.get('verified_email', False),
            raw_data=data
        )


class GitHubOAuth(OAuthProvider):
    """GitHub OAuth 2.0 provider"""
    
    def __init__(self, client_id: str, client_secret: str):
        config = OAuthConfig(
            client_id=client_id,
            client_secret=client_secret,
            authorize_url='https://github.com/login/oauth/authorize',
            token_url='https://github.com/login/oauth/access_token',
            userinfo_url='https://api.github.com/user',
            scopes=['read:user', 'user:email'],
            provider_name='github'
        )
        super().__init__(config)
    
    def get_user_info(self, access_token: str) -> OAuthUserInfo:
        """Fetch GitHub user information"""
        headers = {
            'Authorization': f'Bearer {access_token}',
            'Accept': 'application/json'
        }
        
        # Get user profile
        user_response = self.http_client.get(
            self.config.userinfo_url,
            headers=headers
        )
        user_response.raise_for_status()
        user_data = user_response.json()
        
        # Get primary email (if not public)
        email = user_data.get('email')
        email_verified = False
        
        if not email:
            emails_response = self.http_client.get(
                'https://api.github.com/user/emails',
                headers=headers
            )
            emails_response.raise_for_status()
            emails = emails_response.json()
            
            # Find primary verified email
            for e in emails:
                if e.get('primary') and e.get('verified'):
                    email = e['email']
                    email_verified = True
                    break
            
            # Fallback to first verified email
            if not email:
                for e in emails:
                    if e.get('verified'):
                        email = e['email']
                        email_verified = True
                        break
        
        if not email:
            raise ValueError("No verified email found for GitHub user")
        
        return OAuthUserInfo(
            subject=str(user_data['id']),
            email=email,
            name=user_data.get('name'),
            avatar_url=user_data.get('avatar_url'),
            email_verified=email_verified,
            raw_data=user_data
        )


class OAuthProviderFactory:
    """Factory for creating OAuth providers"""
    
    @staticmethod
    def create_provider(provider_name: str, client_id: str, 
                       client_secret: str) -> OAuthProvider:
        """Create an OAuth provider instance"""
        providers = {
            'google': GoogleOAuth,
            'github': GitHubOAuth,
        }
        
        provider_class = providers.get(provider_name.lower())
        if not provider_class:
            raise ValueError(f"Unknown OAuth provider: {provider_name}")
        
        return provider_class(client_id, client_secret)
