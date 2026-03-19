"""
License Validation Module for MetaOpticsAI.

Handles:
  - Machine fingerprint generation (CPU + MAC + Disk serial)
  - Local encrypted license file (.lic) storage with Fernet symmetric encryption
  - HMAC-SHA256 signature verification to prevent tampering
  - Online license validation against the FastAPI license server
  - 24-hour validation cache (avoid hitting server on every startup)
  - Grace period for offline mode (3 days)
  - Platform-specific license file paths (Windows/macOS/Linux)
"""

import os
import sys
import json
import hmac
import uuid
import hashlib
import platform
import subprocess
import time
import logging
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass

from cryptography.fernet import Fernet, InvalidToken

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════

APP_NAME = "AIPhotonicsDesign"

# HMAC secret for signature verification (in production, obfuscate or derive at runtime)
_HMAC_SECRET = b"MetaOpticsAI-2026-License-HMAC-Secret-Key-v1"

# Fernet encryption key for .lic file (derive from machine + app constant)
_FERNET_SEED = b"MetaOpticsAI-Fernet-Seed-2026-v1"

# Grace period: how many days the app works offline after last successful validation
OFFLINE_GRACE_DAYS = 3

# Re-validation interval: validate online at most once per this interval
REVALIDATION_HOURS = 24

# License server default URL
DEFAULT_LICENSE_SERVER = "https://license.yourcompany.com"


# ═══════════════════════════════════════════════════════════════════════
# Data Structures
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class LicenseInfo:
    """Decoded license information."""
    license_key: str = ""
    user_email: str = ""
    expiry_date: str = ""           # ISO format
    plan_type: str = ""             # "monthly", "yearly", "lifetime"
    signature: str = ""             # HMAC hex digest
    machine_id: str = ""            # bound machine fingerprint
    last_validated: str = ""        # ISO datetime of last online validation
    last_server_message: str = ""   # message from server

    def to_dict(self) -> dict:
        return {
            "license_key": self.license_key,
            "user_email": self.user_email,
            "expiry_date": self.expiry_date,
            "plan_type": self.plan_type,
            "signature": self.signature,
            "machine_id": self.machine_id,
            "last_validated": self.last_validated,
            "last_server_message": self.last_server_message,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "LicenseInfo":
        return cls(
            license_key=d.get("license_key", ""),
            user_email=d.get("user_email", ""),
            expiry_date=d.get("expiry_date", ""),
            plan_type=d.get("plan_type", ""),
            signature=d.get("signature", ""),
            machine_id=d.get("machine_id", ""),
            last_validated=d.get("last_validated", ""),
            last_server_message=d.get("last_server_message", ""),
        )


@dataclass
class ValidationResult:
    """Result of a license validation attempt."""
    is_valid: bool
    message: str
    license_info: Optional[LicenseInfo] = None
    days_remaining: int = 0
    needs_activation: bool = False    # True if no local license exists


# ═══════════════════════════════════════════════════════════════════════
# LicenseValidator
# ═══════════════════════════════════════════════════════════════════════

class LicenseValidator:
    """
    Client-side license validation for MetaOpticsAI.

    Usage:
        validator = LicenseValidator(
            license_server_url="https://license.yourcompany.com",
            app_version="1.0.0"
        )
        result = validator.validate()
        if not result.is_valid:
            # Show activation dialog or error
            ...
    """

    def __init__(self, license_server_url: str = DEFAULT_LICENSE_SERVER,
                 app_version: str = "1.0.0"):
        self.server_url = license_server_url.rstrip("/")
        self.app_version = app_version
        self._license_path = self._get_license_path()
        self._fernet = self._derive_fernet_key()
        self._current_license: Optional[LicenseInfo] = None

    # ─── Platform-specific paths ─────────────────────────────────────

    @staticmethod
    def _get_license_path() -> Path:
        """Get platform-specific license file path."""
        system = platform.system()
        if system == "Windows":
            base = Path(os.environ.get("PROGRAMDATA", "C:\\ProgramData"))
            return base / APP_NAME / "license.lic"
        elif system == "Darwin":
            return Path.home() / "Library" / "Application Support" / APP_NAME / "license.lic"
        else:  # Linux and others
            return Path.home() / ".config" / APP_NAME / "license.lic"

    # ─── Machine fingerprint ─────────────────────────────────────────

    @staticmethod
    def get_machine_id() -> str:
        """
        Generate a unique machine fingerprint from:
          - CPU info (processor string or model name)
          - MAC address (primary NIC)
          - Disk serial number (root volume, where available)

        Returns a deterministic SHA-256 hex digest.
        """
        components = []

        # 1. CPU identifier
        try:
            cpu = platform.processor()
            if not cpu:
                cpu = platform.machine()
            components.append(f"cpu:{cpu}")
        except Exception:
            components.append("cpu:unknown")

        # 2. MAC address (primary interface)
        try:
            mac = uuid.getnode()
            # uuid.getnode() returns a random MAC if it can't find one;
            # check the multicast bit (bit 0 of first octet)
            if (mac >> 40) % 2 == 0:  # valid (non-random) MAC
                mac_str = ':'.join(f'{(mac >> (8*i)) & 0xFF:02x}' for i in range(5, -1, -1))
                components.append(f"mac:{mac_str}")
            else:
                components.append(f"mac:{mac}")
        except Exception:
            components.append("mac:unknown")

        # 3. Disk serial (best-effort)
        try:
            system = platform.system()
            if system == "Windows":
                output = subprocess.check_output(
                    "wmic diskdrive get SerialNumber",
                    shell=True, text=True, timeout=5
                ).strip()
                lines = [l.strip() for l in output.split('\n') if l.strip() and l.strip() != 'SerialNumber']
                if lines:
                    components.append(f"disk:{lines[0]}")
            elif system == "Linux":
                # Try /etc/machine-id first (persistent, standard)
                mid_path = Path("/etc/machine-id")
                if mid_path.exists():
                    components.append(f"disk:{mid_path.read_text().strip()}")
                else:
                    output = subprocess.check_output(
                        ["lsblk", "-ndo", "SERIAL", "/dev/sda"],
                        text=True, timeout=5
                    ).strip()
                    if output:
                        components.append(f"disk:{output}")
            elif system == "Darwin":
                output = subprocess.check_output(
                    ["ioreg", "-rd1", "-c", "IOPlatformExpertDevice"],
                    text=True, timeout=5
                )
                for line in output.split('\n'):
                    if 'IOPlatformSerialNumber' in line:
                        serial = line.split('"')[-2]
                        components.append(f"disk:{serial}")
                        break
        except Exception:
            components.append("disk:fallback")

        # Hash the concatenated components
        combined = "|".join(sorted(components))
        return hashlib.sha256(combined.encode()).hexdigest()

    # ─── Encryption ──────────────────────────────────────────────────

    def _derive_fernet_key(self) -> Fernet:
        """Derive a Fernet key from machine ID + app seed for license file encryption."""
        machine_id = self.get_machine_id()
        key_material = _FERNET_SEED + machine_id.encode()
        # Fernet needs a url-safe base64-encoded 32-byte key
        import base64
        derived = hashlib.sha256(key_material).digest()
        fernet_key = base64.urlsafe_b64encode(derived)
        return Fernet(fernet_key)

    def _encrypt(self, data: dict) -> bytes:
        """Encrypt a dict to bytes."""
        json_bytes = json.dumps(data, indent=None).encode("utf-8")
        return self._fernet.encrypt(json_bytes)

    def _decrypt(self, encrypted: bytes) -> dict:
        """Decrypt bytes to dict. Raises InvalidToken if tampered or wrong machine."""
        json_bytes = self._fernet.decrypt(encrypted)
        return json.loads(json_bytes.decode("utf-8"))

    # ─── HMAC signature ──────────────────────────────────────────────

    @staticmethod
    def _compute_signature(license_key: str, user_email: str,
                           expiry_date: str, plan_type: str) -> str:
        """Compute HMAC-SHA256 signature over license fields."""
        message = f"{license_key}|{user_email}|{expiry_date}|{plan_type}"
        return hmac.new(_HMAC_SECRET, message.encode(), hashlib.sha256).hexdigest()

    @staticmethod
    def _verify_signature(info: LicenseInfo) -> bool:
        """Verify the HMAC signature of a LicenseInfo."""
        expected = LicenseValidator._compute_signature(
            info.license_key, info.user_email,
            info.expiry_date, info.plan_type
        )
        return hmac.compare_digest(info.signature, expected)

    # ─── Local license file I/O ──────────────────────────────────────

    def _read_local_license(self) -> Optional[LicenseInfo]:
        """Read and decrypt local .lic file."""
        if not self._license_path.exists():
            return None
        try:
            encrypted = self._license_path.read_bytes()
            data = self._decrypt(encrypted)
            info = LicenseInfo.from_dict(data)
            return info
        except (InvalidToken, json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to read local license: {e}")
            return None

    def _write_local_license(self, info: LicenseInfo):
        """Encrypt and write license to .lic file."""
        self._license_path.parent.mkdir(parents=True, exist_ok=True)
        encrypted = self._encrypt(info.to_dict())
        self._license_path.write_bytes(encrypted)
        logger.info(f"License saved to {self._license_path}")

    # ─── Expiry checks ───────────────────────────────────────────────

    @staticmethod
    def _parse_expiry(expiry_str: str) -> Optional[datetime]:
        """Parse ISO format expiry date."""
        if not expiry_str:
            return None
        try:
            dt = datetime.fromisoformat(expiry_str.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except (ValueError, TypeError):
            return None

    def is_expired(self, info: Optional[LicenseInfo] = None) -> bool:
        """Check if the license is expired."""
        info = info or self._current_license
        if not info:
            return True
        expiry = self._parse_expiry(info.expiry_date)
        if not expiry:
            return True
        return datetime.now(timezone.utc) > expiry

    def get_days_remaining(self, info: Optional[LicenseInfo] = None) -> int:
        """Calculate days until license expiry."""
        info = info or self._current_license
        if not info:
            return 0
        expiry = self._parse_expiry(info.expiry_date)
        if not expiry:
            return 0
        delta = expiry - datetime.now(timezone.utc)
        return max(0, delta.days)

    # ─── Online validation ───────────────────────────────────────────

    def validate_online(self, license_key: str) -> Tuple[bool, Dict[str, Any]]:
        """
        Contact the license server to validate the key.

        API: GET {server}/api/license/validate
        Headers:
            Authorization: Bearer {license_key}
            X-Machine-ID: {machine_fingerprint}
            X-App-Version: {version}

        Returns:
            (success, response_dict)
        """
        import urllib.request
        import urllib.error

        url = f"{self.server_url}/api/license/validate"
        machine_id = self.get_machine_id()

        headers = {
            "Authorization": f"Bearer {license_key}",
            "X-Machine-ID": machine_id,
            "X-App-Version": self.app_version,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        try:
            req = urllib.request.Request(url, headers=headers, method="GET")
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                is_valid = data.get("valid", False)
                return is_valid, data
        except urllib.error.HTTPError as e:
            try:
                body = json.loads(e.read().decode("utf-8"))
                msg = body.get("message", f"HTTP {e.code}")
            except Exception:
                msg = f"Server error (HTTP {e.code})"
            return False, {"valid": False, "message": msg}
        except urllib.error.URLError as e:
            return False, {"valid": False, "message": f"Cannot reach license server: {e.reason}"}
        except Exception as e:
            return False, {"valid": False, "message": f"Network error: {str(e)}"}

    # ─── Local validation ────────────────────────────────────────────

    def check_local_license(self) -> ValidationResult:
        """
        Check if a valid local license file exists.

        Validates:
          1. File exists and decrypts successfully
          2. Machine ID matches (license bound to this machine)
          3. HMAC signature is valid (not tampered)
          4. License has not expired
          5. Last online validation within grace period
        """
        info = self._read_local_license()

        if info is None:
            return ValidationResult(
                is_valid=False,
                message="No license file found. Please activate your license.",
                needs_activation=True,
            )

        self._current_license = info

        # Check machine binding
        current_machine = self.get_machine_id()
        if info.machine_id and info.machine_id != current_machine:
            return ValidationResult(
                is_valid=False,
                message="License is not registered to this machine. "
                        "Please contact support or re-activate.",
                license_info=info,
            )

        # Verify HMAC signature
        if info.signature and not self._verify_signature(info):
            return ValidationResult(
                is_valid=False,
                message="License file integrity check failed. "
                        "The file may have been tampered with.",
                license_info=info,
            )

        # Check expiry
        if self.is_expired(info):
            days = self.get_days_remaining(info)
            return ValidationResult(
                is_valid=False,
                message=f"Your {info.plan_type} subscription has expired. "
                        f"Please renew to continue using MetaOpticsAI.",
                license_info=info,
                days_remaining=0,
            )

        # Check if online re-validation is needed
        days_remaining = self.get_days_remaining(info)
        return ValidationResult(
            is_valid=True,
            message=f"License valid. {days_remaining} days remaining ({info.plan_type}).",
            license_info=info,
            days_remaining=days_remaining,
        )

    # ─── Needs online re-validation? ─────────────────────────────────

    def _needs_online_revalidation(self, info: LicenseInfo) -> bool:
        """Check if we need to contact the server (every REVALIDATION_HOURS)."""
        if not info.last_validated:
            return True
        last = self._parse_expiry(info.last_validated)
        if not last:
            return True
        elapsed = datetime.now(timezone.utc) - last
        return elapsed > timedelta(hours=REVALIDATION_HOURS)

    def _within_grace_period(self, info: LicenseInfo) -> bool:
        """Check if we're within the offline grace period."""
        if not info.last_validated:
            return False
        last = self._parse_expiry(info.last_validated)
        if not last:
            return False
        elapsed = datetime.now(timezone.utc) - last
        return elapsed <= timedelta(days=OFFLINE_GRACE_DAYS)

    # ─── Save license (activation) ───────────────────────────────────

    def save_license(self, license_key: str, user_email: str = "",
                     expiry_date: str = "", plan_type: str = "",
                     server_response: Optional[dict] = None) -> LicenseInfo:
        """
        Save a new license after successful activation.

        Can be called with explicit fields or with a server_response dict
        that contains expiry_date, plan_type, etc.
        """
        if server_response:
            expiry_date = server_response.get("expiry_date", expiry_date)
            plan_type = server_response.get("plan_type", plan_type)
            user_email = server_response.get("user_email", user_email)

        machine_id = self.get_machine_id()
        signature = self._compute_signature(license_key, user_email, expiry_date, plan_type)
        now_iso = datetime.now(timezone.utc).isoformat()

        info = LicenseInfo(
            license_key=license_key,
            user_email=user_email,
            expiry_date=expiry_date,
            plan_type=plan_type,
            signature=signature,
            machine_id=machine_id,
            last_validated=now_iso,
            last_server_message="Activated successfully",
        )

        self._write_local_license(info)
        self._current_license = info
        return info

    # ─── Main validation entry point ─────────────────────────────────

    def validate(self) -> ValidationResult:
        """
        Main validation method. Called on application startup.

        Flow:
          1. Check local license file
          2. If no local file → needs activation
          3. If local file valid:
             a. If online re-validation needed (>24h since last check):
                - Try online validation
                - If online OK → update local file, allow
                - If online fails → check grace period
                  - Within grace → allow with warning
                  - Outside grace → block
             b. If no re-validation needed → allow
          4. Return ValidationResult
        """
        # Step 1: Local check
        local_result = self.check_local_license()

        # Step 2: No license file at all
        if local_result.needs_activation:
            return local_result

        # Step 3: Local file invalid (expired, tampered, wrong machine)
        if not local_result.is_valid:
            return local_result

        info = local_result.license_info

        # Step 4: Check if online re-validation is needed
        if not self._needs_online_revalidation(info):
            # Recent validation still fresh, allow
            return local_result

        # Step 5: Perform online validation
        online_ok, server_data = self.validate_online(info.license_key)

        if online_ok:
            # Update local license with fresh server data
            info.last_validated = datetime.now(timezone.utc).isoformat()
            info.last_server_message = server_data.get("message", "Valid")
            if server_data.get("expiry_date"):
                info.expiry_date = server_data["expiry_date"]
            if server_data.get("plan_type"):
                info.plan_type = server_data["plan_type"]
            # Re-compute signature with possibly updated fields
            info.signature = self._compute_signature(
                info.license_key, info.user_email,
                info.expiry_date, info.plan_type
            )
            self._write_local_license(info)
            self._current_license = info

            days = server_data.get("days_remaining", self.get_days_remaining(info))
            return ValidationResult(
                is_valid=True,
                message=f"License verified. {days} days remaining ({info.plan_type}).",
                license_info=info,
                days_remaining=days,
            )

        # Step 6: Online validation failed — check grace period
        server_msg = server_data.get("message", "Validation failed")

        if self._within_grace_period(info):
            days = self.get_days_remaining(info)
            return ValidationResult(
                is_valid=True,
                message=f"Offline mode — could not reach server. "
                        f"Grace period active ({OFFLINE_GRACE_DAYS} days). "
                        f"{days} days remaining.",
                license_info=info,
                days_remaining=days,
            )

        # Outside grace period — block
        return ValidationResult(
            is_valid=False,
            message=f"License validation failed: {server_msg}. "
                    f"Please connect to the internet or contact support.",
            license_info=info,
            days_remaining=0,
        )

    # ─── Activate with key (for activation dialog) ───────────────────

    def activate(self, license_key: str, email: str = "") -> ValidationResult:
        """
        Activate a new license key. Called from the activation dialog.

        1. Validate the key online
        2. If valid, save to local file
        3. Return result
        """
        online_ok, server_data = self.validate_online(license_key)

        if not online_ok:
            msg = server_data.get("message", "Invalid license key")
            return ValidationResult(
                is_valid=False,
                message=f"Activation failed: {msg}",
                needs_activation=True,
            )

        # Save license locally
        info = self.save_license(
            license_key=license_key,
            user_email=email or server_data.get("user_email", ""),
            server_response=server_data,
        )

        days = server_data.get("days_remaining", self.get_days_remaining(info))
        return ValidationResult(
            is_valid=True,
            message=f"License activated successfully! "
                    f"{days} days remaining ({info.plan_type}).",
            license_info=info,
            days_remaining=days,
        )

    # ─── Deactivate (remove local license) ───────────────────────────

    def deactivate(self) -> bool:
        """Remove the local license file."""
        try:
            if self._license_path.exists():
                self._license_path.unlink()
            self._current_license = None
            return True
        except Exception as e:
            logger.error(f"Failed to deactivate: {e}")
            return False

    # ─── Properties ──────────────────────────────────────────────────

    @property
    def license_file_path(self) -> Path:
        return self._license_path

    @property
    def current_license(self) -> Optional[LicenseInfo]:
        return self._current_license

    @property
    def is_activated(self) -> bool:
        return self._license_path.exists()