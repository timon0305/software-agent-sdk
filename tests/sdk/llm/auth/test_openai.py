"""Tests for OpenAI subscription authentication."""

import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from openhands.sdk.llm.auth.credentials import CredentialStore, OAuthCredentials
from openhands.sdk.llm.auth.openai import (
    CLIENT_ID,
    CONSENT_BANNER,
    CONSENT_DISCLAIMER,
    ISSUER,
    OPENAI_CODEX_MODELS,
    OpenAISubscriptionAuth,
    _build_authorize_url,
    _display_consent_and_confirm,
    _extract_chatgpt_account_id,
    _generate_pkce,
    _get_consent_marker_path,
    _has_acknowledged_consent,
    _jwks_cache,
    _mark_consent_acknowledged,
)


def test_generate_pkce():
    """Test PKCE code generation using authlib."""
    verifier, challenge = _generate_pkce()
    assert verifier is not None
    assert challenge is not None
    assert len(verifier) > 0
    assert len(challenge) > 0
    # Verifier and challenge should be different
    assert verifier != challenge


def test_pkce_codes_are_unique():
    """Test that PKCE codes are unique each time."""
    verifier1, challenge1 = _generate_pkce()
    verifier2, challenge2 = _generate_pkce()
    assert verifier1 != verifier2
    assert challenge1 != challenge2


def test_build_authorize_url():
    """Test building the OAuth authorization URL."""
    code_challenge = "test_challenge"
    state = "test_state"
    redirect_uri = "http://localhost:1455/auth/callback"

    url = _build_authorize_url(redirect_uri, code_challenge, state)

    assert url.startswith(f"{ISSUER}/oauth/authorize?")
    assert f"client_id={CLIENT_ID}" in url
    assert "redirect_uri=http%3A%2F%2Flocalhost%3A1455%2Fauth%2Fcallback" in url
    assert "code_challenge=test_challenge" in url
    assert "code_challenge_method=S256" in url
    assert "state=test_state" in url
    assert "originator=openhands" in url
    assert "response_type=code" in url


def test_openai_codex_models():
    """Test that OPENAI_CODEX_MODELS contains expected models."""
    assert "gpt-5.2-codex" in OPENAI_CODEX_MODELS
    assert "gpt-5.2" in OPENAI_CODEX_MODELS
    assert "gpt-5.1-codex-max" in OPENAI_CODEX_MODELS
    assert "gpt-5.1-codex-mini" in OPENAI_CODEX_MODELS


def test_openai_subscription_auth_vendor():
    """Test OpenAISubscriptionAuth vendor property."""
    auth = OpenAISubscriptionAuth()
    assert auth.vendor == "openai"


def test_openai_subscription_auth_get_credentials(tmp_path):
    """Test getting credentials from store."""
    store = CredentialStore(credentials_dir=tmp_path)
    auth = OpenAISubscriptionAuth(credential_store=store)

    # No credentials initially
    assert auth.get_credentials() is None

    # Save credentials
    creds = OAuthCredentials(
        vendor="openai",
        access_token="test_access",
        refresh_token="test_refresh",
        expires_at=int(time.time() * 1000) + 3600_000,
    )
    store.save(creds)

    # Now should return credentials
    retrieved = auth.get_credentials()
    assert retrieved is not None
    assert retrieved.access_token == "test_access"


def test_openai_subscription_auth_has_valid_credentials(tmp_path):
    """Test checking for valid credentials."""
    store = CredentialStore(credentials_dir=tmp_path)
    auth = OpenAISubscriptionAuth(credential_store=store)

    # No credentials
    assert not auth.has_valid_credentials()

    # Valid credentials
    valid_creds = OAuthCredentials(
        vendor="openai",
        access_token="test",
        refresh_token="test",
        expires_at=int(time.time() * 1000) + 3600_000,
    )
    store.save(valid_creds)
    assert auth.has_valid_credentials()

    # Expired credentials
    expired_creds = OAuthCredentials(
        vendor="openai",
        access_token="test",
        refresh_token="test",
        expires_at=int(time.time() * 1000) - 3600_000,
    )
    store.save(expired_creds)
    assert not auth.has_valid_credentials()


def test_openai_subscription_auth_logout(tmp_path):
    """Test logout removes credentials."""
    store = CredentialStore(credentials_dir=tmp_path)
    auth = OpenAISubscriptionAuth(credential_store=store)

    # Save credentials
    creds = OAuthCredentials(
        vendor="openai",
        access_token="test",
        refresh_token="test",
        expires_at=int(time.time() * 1000) + 3600_000,
    )
    store.save(creds)
    assert auth.has_valid_credentials()

    # Logout
    assert auth.logout() is True
    assert not auth.has_valid_credentials()

    # Logout again should return False
    assert auth.logout() is False


def test_openai_subscription_auth_create_llm_invalid_model(tmp_path):
    """Test create_llm raises error for invalid model."""
    store = CredentialStore(credentials_dir=tmp_path)
    auth = OpenAISubscriptionAuth(credential_store=store)

    # Save valid credentials
    creds = OAuthCredentials(
        vendor="openai",
        access_token="test",
        refresh_token="test",
        expires_at=int(time.time() * 1000) + 3600_000,
    )
    store.save(creds)

    with pytest.raises(ValueError, match="not supported for subscription access"):
        auth.create_llm(model="gpt-4")


def test_openai_subscription_auth_create_llm_no_credentials(tmp_path):
    """Test create_llm raises error when no credentials available."""
    store = CredentialStore(credentials_dir=tmp_path)
    auth = OpenAISubscriptionAuth(credential_store=store)

    with pytest.raises(ValueError, match="No credentials available"):
        auth.create_llm(model="gpt-5.2-codex")


def test_openai_subscription_auth_create_llm_success(tmp_path):
    """Test create_llm creates LLM with correct configuration."""
    store = CredentialStore(credentials_dir=tmp_path)
    auth = OpenAISubscriptionAuth(credential_store=store)

    # Save valid credentials
    creds = OAuthCredentials(
        vendor="openai",
        access_token="test_access_token",
        refresh_token="test_refresh",
        expires_at=int(time.time() * 1000) + 3600_000,
    )
    store.save(creds)

    llm = auth.create_llm(model="gpt-5.2-codex")

    assert llm.model == "openai/gpt-5.2-codex"
    assert llm.api_key is not None
    assert llm.extra_headers is not None
    # Uses codex_cli_rs to match official Codex CLI for compatibility
    assert llm.extra_headers.get("originator") == "codex_cli_rs"


@pytest.mark.asyncio
async def test_openai_subscription_auth_refresh_if_needed_no_creds(tmp_path):
    """Test refresh_if_needed returns None when no credentials."""
    store = CredentialStore(credentials_dir=tmp_path)
    auth = OpenAISubscriptionAuth(credential_store=store)

    result = await auth.refresh_if_needed()
    assert result is None


@pytest.mark.asyncio
async def test_openai_subscription_auth_refresh_if_needed_valid_creds(tmp_path):
    """Test refresh_if_needed returns existing creds when not expired."""
    store = CredentialStore(credentials_dir=tmp_path)
    auth = OpenAISubscriptionAuth(credential_store=store)

    # Save valid credentials
    creds = OAuthCredentials(
        vendor="openai",
        access_token="test_access",
        refresh_token="test_refresh",
        expires_at=int(time.time() * 1000) + 3600_000,
    )
    store.save(creds)

    result = await auth.refresh_if_needed()
    assert result is not None
    assert result.access_token == "test_access"


@pytest.mark.asyncio
async def test_openai_subscription_auth_refresh_if_needed_expired_creds(tmp_path):
    """Test refresh_if_needed refreshes expired credentials."""
    store = CredentialStore(credentials_dir=tmp_path)
    auth = OpenAISubscriptionAuth(credential_store=store)

    # Save expired credentials
    creds = OAuthCredentials(
        vendor="openai",
        access_token="old_access",
        refresh_token="test_refresh",
        expires_at=int(time.time() * 1000) - 3600_000,
    )
    store.save(creds)

    # Mock the refresh function
    with patch(
        "openhands.sdk.llm.auth.openai._refresh_access_token",
        new_callable=AsyncMock,
    ) as mock_refresh:
        mock_refresh.return_value = {
            "access_token": "new_access",
            "refresh_token": "new_refresh",
            "expires_in": 3600,
        }

        result = await auth.refresh_if_needed()

        assert result is not None
        assert result.access_token == "new_access"
        mock_refresh.assert_called_once_with("test_refresh")


# Tests for JWT signature verification
class TestJWTVerification:
    """Tests for JWT signature verification in _extract_chatgpt_account_id."""

    def setup_method(self):
        """Clear JWKS cache before each test."""
        _jwks_cache.clear()

    def test_extract_chatgpt_account_id_with_valid_jwt(self):
        """Test extracting account ID from a properly signed JWT."""
        # Mock JWKS and JWT verification
        mock_claims = MagicMock()
        mock_claims.get.return_value = {"chatgpt_account_id": "test-account-123"}

        with (
            patch.object(_jwks_cache, "get_key_set") as mock_get_keys,
            patch("openhands.sdk.llm.auth.openai.jwt.decode") as mock_decode,
        ):
            mock_get_keys.return_value = MagicMock()
            mock_decode.return_value = mock_claims

            result = _extract_chatgpt_account_id("valid.jwt.token")

            assert result == "test-account-123"
            mock_decode.assert_called_once()
            mock_claims.validate.assert_called_once()

    def test_extract_chatgpt_account_id_invalid_signature(self):
        """Test that invalid JWT signature returns None."""
        from authlib.jose.errors import DecodeError

        with (
            patch.object(_jwks_cache, "get_key_set") as mock_get_keys,
            patch("openhands.sdk.llm.auth.openai.jwt.decode") as mock_decode,
        ):
            mock_get_keys.return_value = MagicMock()
            mock_decode.side_effect = DecodeError("Invalid signature")

            result = _extract_chatgpt_account_id("invalid.signature.token")

            assert result is None

    def test_extract_chatgpt_account_id_jwks_fetch_failure(self):
        """Test graceful handling when JWKS cannot be fetched."""
        with patch.object(_jwks_cache, "get_key_set") as mock_get_keys:
            mock_get_keys.side_effect = RuntimeError("Network error")

            result = _extract_chatgpt_account_id("some.jwt.token")

            assert result is None

    def test_extract_chatgpt_account_id_missing_account_id(self):
        """Test handling JWT without chatgpt_account_id claim."""
        mock_claims = MagicMock()
        mock_claims.get.return_value = {}  # No chatgpt_account_id

        with (
            patch.object(_jwks_cache, "get_key_set") as mock_get_keys,
            patch("openhands.sdk.llm.auth.openai.jwt.decode") as mock_decode,
        ):
            mock_get_keys.return_value = MagicMock()
            mock_decode.return_value = mock_claims

            result = _extract_chatgpt_account_id("valid.jwt.token")

            assert result is None

    def test_extract_chatgpt_account_id_expired_token(self):
        """Test handling expired JWT."""
        from authlib.jose.errors import ExpiredTokenError

        mock_claims = MagicMock()
        mock_claims.validate.side_effect = ExpiredTokenError()

        with (
            patch.object(_jwks_cache, "get_key_set") as mock_get_keys,
            patch("openhands.sdk.llm.auth.openai.jwt.decode") as mock_decode,
        ):
            mock_get_keys.return_value = MagicMock()
            mock_decode.return_value = mock_claims

            result = _extract_chatgpt_account_id("expired.jwt.token")

            assert result is None


class TestJWKSCache:
    """Tests for the JWKS cache."""

    def setup_method(self):
        """Clear JWKS cache before each test."""
        _jwks_cache.clear()

    def test_jwks_cache_fetches_on_first_call(self):
        """Test that JWKS is fetched on first call."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "keys": [{"kid": "test-key", "kty": "RSA", "n": "abc", "e": "AQAB"}]
        }

        with patch("openhands.sdk.llm.auth.openai.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.get.return_value = mock_response
            mock_client_class.return_value = mock_client

            # This should trigger a fetch
            _jwks_cache.get_key_set()

            mock_client.get.assert_called_once()

    def test_jwks_cache_uses_cached_value(self):
        """Test that cached JWKS is reused within TTL."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "keys": [{"kid": "test-key", "kty": "RSA", "n": "abc", "e": "AQAB"}]
        }

        with patch("openhands.sdk.llm.auth.openai.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.get.return_value = mock_response
            mock_client_class.return_value = mock_client

            # First call fetches
            _jwks_cache.get_key_set()
            # Second call should use cache
            _jwks_cache.get_key_set()

            # Should only fetch once
            assert mock_client.get.call_count == 1

    def test_jwks_cache_clear(self):
        """Test that cache can be cleared."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "keys": [{"kid": "test-key", "kty": "RSA", "n": "abc", "e": "AQAB"}]
        }

        with patch("openhands.sdk.llm.auth.openai.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.get.return_value = mock_response
            mock_client_class.return_value = mock_client

            # First call fetches
            _jwks_cache.get_key_set()
            # Clear cache
            _jwks_cache.clear()
            # Next call should fetch again
            _jwks_cache.get_key_set()

            assert mock_client.get.call_count == 2


# =========================================================================
# Tests for consent banner system
# =========================================================================


class TestConsentBannerSystem:
    """Tests for the consent banner and acknowledgment system."""

    def test_consent_banner_content(self):
        """Test that consent banner contains required text."""
        assert "ChatGPT" in CONSENT_BANNER
        assert "Terms of Use" in CONSENT_BANNER
        assert "openai.com/policies/terms-of-use" in CONSENT_BANNER

    def test_consent_disclaimer_content(self):
        """Test that consent disclaimer contains required text."""
        assert "Data rights & privacy" in CONSENT_DISCLAIMER
        assert "Prohibited data" in CONSENT_DISCLAIMER
        assert "Protected Health Information" in CONSENT_DISCLAIMER
        assert "children under 13" in CONSENT_DISCLAIMER

    def test_consent_marker_path(self, tmp_path):
        """Test that consent marker path is in credentials directory."""
        with patch(
            "openhands.sdk.llm.auth.openai.get_credentials_dir", return_value=tmp_path
        ):
            marker_path = _get_consent_marker_path()
            assert marker_path.parent == tmp_path
            assert ".chatgpt_consent_acknowledged" in str(marker_path)

    def test_has_acknowledged_consent_false_initially(self, tmp_path):
        """Test that consent is not acknowledged initially."""
        with patch(
            "openhands.sdk.llm.auth.openai.get_credentials_dir", return_value=tmp_path
        ):
            assert not _has_acknowledged_consent()

    def test_mark_consent_acknowledged(self, tmp_path):
        """Test marking consent as acknowledged."""
        with patch(
            "openhands.sdk.llm.auth.openai.get_credentials_dir", return_value=tmp_path
        ):
            assert not _has_acknowledged_consent()
            _mark_consent_acknowledged()
            assert _has_acknowledged_consent()

    def test_display_consent_user_accepts(self, tmp_path, capsys):
        """Test consent display when user accepts."""
        with (
            patch(
                "openhands.sdk.llm.auth.openai.get_credentials_dir",
                return_value=tmp_path,
            ),
            patch("sys.stdin.isatty", return_value=True),
            patch("builtins.input", return_value="y"),
        ):
            result = _display_consent_and_confirm(verbose=False)
            assert result is True

            # Check banner was printed
            captured = capsys.readouterr()
            assert "ChatGPT" in captured.out
            assert "Terms of Use" in captured.out

    def test_display_consent_user_declines(self, tmp_path, capsys):
        """Test consent display when user declines."""
        with (
            patch(
                "openhands.sdk.llm.auth.openai.get_credentials_dir",
                return_value=tmp_path,
            ),
            patch("sys.stdin.isatty", return_value=True),
            patch("builtins.input", return_value="n"),
        ):
            result = _display_consent_and_confirm(verbose=False)
            assert result is False

    def test_display_consent_first_time_shows_disclaimer(self, tmp_path, capsys):
        """Test that first-time consent shows full disclaimer."""
        with (
            patch(
                "openhands.sdk.llm.auth.openai.get_credentials_dir",
                return_value=tmp_path,
            ),
            patch("sys.stdin.isatty", return_value=True),
            patch("builtins.input", return_value="y"),
        ):
            _display_consent_and_confirm(verbose=False)
            captured = capsys.readouterr()
            # First time should show disclaimer
            assert "Data rights & privacy" in captured.out

    def test_display_consent_subsequent_time_no_disclaimer(self, tmp_path, capsys):
        """Test that subsequent consent does not show full disclaimer."""
        with (
            patch(
                "openhands.sdk.llm.auth.openai.get_credentials_dir",
                return_value=tmp_path,
            ),
            patch("sys.stdin.isatty", return_value=True),
            patch("builtins.input", return_value="y"),
        ):
            # First time - acknowledge
            _display_consent_and_confirm(verbose=False)
            capsys.readouterr()  # Clear output

            # Second time - should not show disclaimer
            _display_consent_and_confirm(verbose=False)
            captured = capsys.readouterr()
            # Banner should still be shown
            assert "ChatGPT" in captured.out
            # But not the full disclaimer
            assert "Data rights & privacy" not in captured.out

    def test_display_consent_verbose_always_shows_disclaimer(self, tmp_path, capsys):
        """Test that verbose flag always shows full disclaimer."""
        with (
            patch(
                "openhands.sdk.llm.auth.openai.get_credentials_dir",
                return_value=tmp_path,
            ),
            patch("sys.stdin.isatty", return_value=True),
            patch("builtins.input", return_value="y"),
        ):
            # First time - acknowledge
            _display_consent_and_confirm(verbose=False)
            capsys.readouterr()  # Clear output

            # Second time with verbose - should show disclaimer
            _display_consent_and_confirm(verbose=True)
            captured = capsys.readouterr()
            assert "Data rights & privacy" in captured.out

    def test_display_consent_non_interactive_first_time_raises(self, tmp_path):
        """Test that non-interactive mode raises error on first time."""
        with (
            patch(
                "openhands.sdk.llm.auth.openai.get_credentials_dir",
                return_value=tmp_path,
            ),
            patch("sys.stdin.isatty", return_value=False),
        ):
            with pytest.raises(RuntimeError, match="non-interactive mode"):
                _display_consent_and_confirm(verbose=False)

    def test_display_consent_non_interactive_after_acknowledgment(self, tmp_path):
        """Test that non-interactive mode works after prior acknowledgment."""
        with patch(
            "openhands.sdk.llm.auth.openai.get_credentials_dir", return_value=tmp_path
        ):
            # Mark consent as acknowledged
            _mark_consent_acknowledged()

            with patch("sys.stdin.isatty", return_value=False):
                result = _display_consent_and_confirm(verbose=False)
                assert result is True

    def test_display_consent_keyboard_interrupt(self, tmp_path):
        """Test handling of keyboard interrupt during consent."""
        with (
            patch(
                "openhands.sdk.llm.auth.openai.get_credentials_dir",
                return_value=tmp_path,
            ),
            patch("sys.stdin.isatty", return_value=True),
            patch("builtins.input", side_effect=KeyboardInterrupt),
        ):
            result = _display_consent_and_confirm(verbose=False)
            assert result is False

    def test_display_consent_eof_error(self, tmp_path):
        """Test handling of EOF during consent."""
        with (
            patch(
                "openhands.sdk.llm.auth.openai.get_credentials_dir",
                return_value=tmp_path,
            ),
            patch("sys.stdin.isatty", return_value=True),
            patch("builtins.input", side_effect=EOFError),
        ):
            result = _display_consent_and_confirm(verbose=False)
            assert result is False
