from __future__ import annotations

import hashlib
import os
from pathlib import Path

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

MAGIC = b"VLPTEST1"
SALT_BYTES = 16
NONCE_BYTES = 12
KEY_BYTES = 32
SCRYPT_N = 2**15
SCRYPT_R = 8
SCRYPT_P = 1


def _derive_key(password: str, salt: bytes) -> bytes:
    if not password:
        raise ValueError("Password must not be empty")
    return hashlib.scrypt(
        password.encode("utf-8"),
        salt=salt,
        n=SCRYPT_N,
        r=SCRYPT_R,
        p=SCRYPT_P,
        dklen=KEY_BYTES,
        maxmem=128 * 1024 * 1024,
    )


def encrypt_bytes(plaintext: bytes, password: str) -> bytes:
    """Encrypt bytes using AES-256-GCM with a scrypt-derived key.

    Format: MAGIC | salt[16] | nonce[12] | AESGCM(ciphertext || tag)
    The header is authenticated as additional data.
    """
    salt = os.urandom(SALT_BYTES)
    nonce = os.urandom(NONCE_BYTES)
    key = _derive_key(password, salt)
    ciphertext = AESGCM(key).encrypt(nonce, plaintext, MAGIC)
    return MAGIC + salt + nonce + ciphertext


def decrypt_bytes(blob: bytes, password: str) -> bytes:
    minimum = len(MAGIC) + SALT_BYTES + NONCE_BYTES + 16
    if len(blob) < minimum:
        raise ValueError("Encrypted test file is truncated")
    if blob[: len(MAGIC)] != MAGIC:
        raise ValueError("Not a VLPTEST1 encrypted file")
    offset = len(MAGIC)
    salt = blob[offset : offset + SALT_BYTES]
    offset += SALT_BYTES
    nonce = blob[offset : offset + NONCE_BYTES]
    offset += NONCE_BYTES
    ciphertext = blob[offset:]
    key = _derive_key(password, salt)
    try:
        return AESGCM(key).decrypt(nonce, ciphertext, MAGIC)
    except Exception as exc:
        raise ValueError("Could not decrypt hidden test set: wrong password or corrupted file") from exc


def encrypt_file(source: Path, destination: Path, password: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(encrypt_bytes(source.read_bytes(), password))


def decrypt_file_to_bytes(source: Path, password: str) -> bytes:
    return decrypt_bytes(source.read_bytes(), password)
