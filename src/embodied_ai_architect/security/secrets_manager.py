"""Secrets management for secure credential handling."""

import os
import re
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any


class SecretError(Exception):
    """Base exception for secret-related errors."""

    pass


class SecretNotFoundError(SecretError):
    """Raised when a required secret is not found."""

    pass


class SecretsProvider(ABC):
    """Abstract base class for secret providers.

    Implementations can load secrets from various sources:
    - Environment variables
    - Files
    - Cloud secret managers (AWS, GCP, Azure)
    - Vault services (HashiCorp Vault)
    - OS keychains
    """

    @abstractmethod
    def get_secret(self, key: str) -> str | None:
        """Get a secret by key.

        Args:
            key: Secret identifier

        Returns:
            Secret value or None if not found
        """
        pass

    @abstractmethod
    def list_keys(self) -> List[str]:
        """List all available secret keys.

        Returns:
            List of secret key names
        """
        pass


class EnvironmentSecretsProvider(SecretsProvider):
    """Load secrets from environment variables.

    Convention: {PREFIX}{KEY_NAME}
    Example: EMBODIED_AI_SSH_KEY_DEV
    """

    def __init__(self, prefix: str = "EMBODIED_AI_"):
        """Initialize environment secrets provider.

        Args:
            prefix: Prefix for environment variables
        """
        self.prefix = prefix

    def get_secret(self, key: str) -> str | None:
        """Get secret from environment variable.

        Args:
            key: Secret key (will be uppercased and prefixed)

        Returns:
            Secret value or None
        """
        env_key = f"{self.prefix}{key.upper()}"
        return os.getenv(env_key)

    def list_keys(self) -> List[str]:
        """List all secret keys from environment.

        Returns:
            List of secret keys (without prefix, lowercased)
        """
        prefix_len = len(self.prefix)
        return [k[prefix_len:].lower() for k in os.environ if k.startswith(self.prefix)]


class FileSecretsProvider(SecretsProvider):
    """Load secrets from files with security checks.

    Security requirements:
    - Files must have 600 or 400 permissions (not world-readable)
    - Directory must be owned by current user
    - Directory should have 700 permissions
    """

    def __init__(self, secrets_dir: str | Path):
        """Initialize file secrets provider.

        Args:
            secrets_dir: Directory containing secret files

        Raises:
            SecretError: If directory doesn't exist or has insecure permissions
        """
        self.secrets_dir = Path(secrets_dir)
        self._validate_directory()

    def _validate_directory(self):
        """Validate secrets directory security.

        Raises:
            SecretError: If directory is insecure
        """
        if not self.secrets_dir.exists():
            # Directory doesn't exist - this is OK, provider just won't find secrets
            return

        # Check ownership (Unix only)
        if hasattr(os, "getuid"):
            stat = self.secrets_dir.stat()
            if stat.st_uid != os.getuid():
                raise SecretError(f"Secrets directory {self.secrets_dir} not owned by current user")

            # Check permissions - others should have no access
            mode = oct(stat.st_mode)[-3:]
            if int(mode[2]) > 0:  # Others have any permission
                raise SecretError(
                    f"Insecure permissions on {self.secrets_dir}: {mode}. "
                    f"Others should have no access (chmod 700 recommended)"
                )

    def get_secret(self, key: str) -> str | None:
        """Get secret from file.

        File path is derived from key by replacing underscores with slashes.
        Example: "ssh_remote_key" → "ssh/remote/key"

        Args:
            key: Secret key

        Returns:
            Secret value or None

        Raises:
            SecretError: If file has insecure permissions
        """
        # Map key to file path: ssh_remote_key → ssh/remote/key
        file_path = self.secrets_dir / key.replace("_", "/")

        if not file_path.exists():
            return None

        # Security check: file permissions (Unix only)
        if hasattr(os, "getuid"):
            stat = file_path.stat()
            mode = oct(stat.st_mode)[-3:]

            # File should be readable only by owner
            if mode not in ["400", "600"]:
                raise SecretError(
                    f"Insecure file permissions {mode} for {file_path}. "
                    f"File should be 400 (read-only) or 600 (read-write) for owner only. "
                    f"Run: chmod 600 {file_path}"
                )

        return file_path.read_text().strip()

    def list_keys(self) -> List[str]:
        """List all secret files.

        Returns:
            List of secret keys
        """
        if not self.secrets_dir.exists():
            return []

        keys = []
        for file_path in self.secrets_dir.rglob("*"):
            if file_path.is_file():
                # Convert path to key: ssh/remote/key → ssh_remote_key
                relative = file_path.relative_to(self.secrets_dir)
                key = str(relative).replace("/", "_")
                keys.append(key)

        return keys


class SecretsManager:
    """Central secrets management with multi-provider support.

    The SecretsManager tries providers in order until a secret is found.
    This allows for configuration hierarchy (e.g., env vars override files).

    Features:
    - Multi-provider support
    - Audit logging
    - Secret masking in logs
    - Variable substitution in config
    """

    def __init__(self, providers: List[SecretsProvider] | None = None):
        """Initialize secrets manager.

        Args:
            providers: List of secret providers (tried in order)
        """
        if providers is None:
            # Default: try environment first, then files
            providers = [EnvironmentSecretsProvider(), FileSecretsProvider("config/credentials")]

        self.providers = providers
        self._audit_log: List[Dict[str, Any]] = []
        self._secret_cache: Dict[str, str] = {}

    def get_secret(self, key: str, required: bool = True, default: str | None = None) -> str | None:
        """Get a secret from any provider.

        Providers are tried in order until secret is found.

        Args:
            key: Secret key
            required: If True, raise error if not found
            default: Default value if not found and not required

        Returns:
            Secret value

        Raises:
            SecretNotFoundError: If required and not found
        """
        # Check cache first
        if key in self._secret_cache:
            return self._secret_cache[key]

        # Try each provider
        for provider in self.providers:
            try:
                value = provider.get_secret(key)
                if value is not None:
                    # Cache the value
                    self._secret_cache[key] = value

                    # Audit log
                    self._audit_log.append(
                        {
                            "timestamp": datetime.now().isoformat(),
                            "action": "secret_accessed",
                            "key": key,
                            "provider": provider.__class__.__name__,
                            "success": True,
                        }
                    )

                    return value
            except SecretError:
                # Provider failed, try next one
                continue

        # Not found in any provider
        if required and default is None:
            raise SecretNotFoundError(
                f"Required secret '{key}' not found in any provider. "
                f"Tried: {[p.__class__.__name__ for p in self.providers]}"
            )

        return default

    def mask_secret(self, text: str, secret: str | None) -> str:
        """Replace secret in text with [REDACTED].

        Args:
            text: Text that might contain secret
            secret: Secret to mask

        Returns:
            Text with secret masked
        """
        if not secret or not text:
            return text

        return text.replace(secret, "[REDACTED]")

    def mask_all_secrets(self, text: str) -> str:
        """Mask all known secrets in text.

        Args:
            text: Text that might contain secrets

        Returns:
            Text with all secrets masked
        """
        result = text
        for secret in self._secret_cache.values():
            result = self.mask_secret(result, secret)
        return result

    def resolve_references(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve secret references in configuration.

        Supports:
        - ${secret:key_name} - Load from secret manager
        - ${env:VAR_NAME} - Load from environment

        Args:
            config: Configuration dictionary

        Returns:
            Configuration with references resolved
        """
        import json

        # Convert to JSON and back to handle nested structures
        config_str = json.dumps(config)

        # Find all ${...} patterns
        pattern = r"\$\{(secret|env):([^}]+)\}"

        def replace_reference(match):
            ref_type = match.group(1)
            ref_key = match.group(2)

            if ref_type == "secret":
                value = self.get_secret(ref_key, required=True)
            elif ref_type == "env":
                value = os.getenv(ref_key)
                if value is None:
                    raise SecretNotFoundError(f"Environment variable '{ref_key}' not found")
            else:
                return match.group(0)

            return value

        resolved_str = re.sub(pattern, replace_reference, config_str)
        return json.loads(resolved_str)

    def get_audit_log(self) -> List[Dict[str, Any]]:
        """Get audit log of secret accesses.

        Returns:
            List of audit log entries
        """
        return self._audit_log.copy()

    def list_all_keys(self) -> Dict[str, List[str]]:
        """List all available keys from all providers.

        Returns:
            Dictionary mapping provider name to list of keys
        """
        all_keys = {}
        for provider in self.providers:
            provider_name = provider.__class__.__name__
            all_keys[provider_name] = provider.list_keys()
        return all_keys
