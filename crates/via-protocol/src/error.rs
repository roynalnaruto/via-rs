//! Protocol-level error type shared by client, server, and wire format.
//!
//! All fallible protocol operations return `crate::Result<T>` (an alias for
//! `core::result::Result<T, ViaError>`). The `ViaError` enum carries only the
//! discriminant + optional payload needed to let callers branch — it does NOT
//! embed secret material. Primitives keep their `_CHECK` / panic model;
//! `ViaError` is the protocol boundary.

use core::fmt;

/// Errors produced at the VIA-C protocol boundary.
///
/// # Design note
/// The variants are ordered coarsely by lifecycle stage (setup → query →
/// answer → recover) so match arms stay readable. Add new variants at the
/// end of the relevant stage group to preserve ABI stability of existing
/// discriminants during development.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ViaError {
    // ── Setup errors ────────────────────────────────────────────────────────
    /// A database operation was attempted before `setup_db` completed.
    SetupNotComplete,

    // ── Query errors ────────────────────────────────────────────────────────
    /// The requested index is out of range for the current database layout.
    IndexOutOfRange {
        /// The supplied (out-of-range) index.
        index: usize,
        /// The total record count in the database.
        num_records: usize,
    },

    /// The `CompressedQuery.ciphertexts` length does not match what the
    /// server expected for the current `PublicParams` layout.
    QueryLengthMismatch {
        /// The number of ciphertexts the layout requires.
        expected: usize,
        /// The number of ciphertexts actually supplied.
        got: usize,
    },

    // ── Wire-format errors ───────────────────────────────────────────────────
    /// Serialized byte buffer is too short to decode a complete record.
    BufferTooShort {
        /// The number of bytes needed to decode.
        needed: usize,
        /// The number of bytes actually available.
        got: usize,
    },

    /// Serialized bytes contain an unrecognized format tag.
    UnknownFormatTag(u8),

    /// A PRG seed embedded in a `PrgCompressed` packet is invalid (e.g.
    /// the byte layout does not match the pinned format).
    InvalidPrgSeed,

    // ── Internal / misc ──────────────────────────────────────────────────────
    /// A dimension mismatch was detected at the protocol layer that the
    /// const-generic `_CHECK` blocks could not catch (e.g. mismatched
    /// runtime `num_rows` vs. the preset's `N1` const).
    DimMismatch(&'static str),
}

impl fmt::Display for ViaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SetupNotComplete => f.write_str("database setup is not complete"),
            Self::IndexOutOfRange { index, num_records } => {
                write!(
                    f,
                    "index {index} is out of range for a database of {num_records} records"
                )
            }
            Self::QueryLengthMismatch { expected, got } => {
                write!(
                    f,
                    "query length mismatch: expected {expected} ciphertexts, got {got}"
                )
            }
            Self::BufferTooShort { needed, got } => {
                write!(f, "buffer too short: needed {needed} bytes, got {got}")
            }
            Self::UnknownFormatTag(tag) => {
                write!(f, "unknown wire-format tag 0x{tag:02x}")
            }
            Self::InvalidPrgSeed => f.write_str("PRG seed in PrgCompressed packet is invalid"),
            Self::DimMismatch(msg) => write!(f, "dimension mismatch: {msg}"),
        }
    }
}

/// Convenience alias. All protocol operations that can fail return this type.
///
/// ```
/// use via_protocol::{Result, ViaError};
/// fn example() -> Result<u32> {
///     Err(ViaError::SetupNotComplete)
/// }
/// assert!(example().is_err());
/// ```
pub type Result<T> = core::result::Result<T, ViaError>;

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::format;

    #[test]
    fn via_error_display_setup_not_complete() {
        let msg = format!("{}", ViaError::SetupNotComplete);
        assert!(msg.contains("setup"));
    }

    #[test]
    fn via_error_display_index_out_of_range() {
        let e = ViaError::IndexOutOfRange {
            index: 99,
            num_records: 10,
        };
        let msg = format!("{e}");
        assert!(msg.contains("99") && msg.contains("10"));
    }

    #[test]
    fn via_error_display_query_length_mismatch() {
        let e = ViaError::QueryLengthMismatch {
            expected: 8,
            got: 3,
        };
        let msg = format!("{e}");
        assert!(msg.contains('8') && msg.contains('3'));
    }

    #[test]
    fn via_error_display_buffer_too_short() {
        let e = ViaError::BufferTooShort {
            needed: 128,
            got: 64,
        };
        let msg = format!("{e}");
        assert!(msg.contains("128") && msg.contains("64"));
    }

    #[test]
    fn result_alias_is_core_result() {
        let ok: Result<i32> = Ok(42);
        assert_eq!(ok, Ok(42));
        let err: Result<i32> = Err(ViaError::SetupNotComplete);
        assert!(err.is_err());
    }

    #[test]
    fn via_error_is_clone_and_eq() {
        let e1 = ViaError::IndexOutOfRange {
            index: 0,
            num_records: 1,
        };
        let e2 = e1.clone();
        assert_eq!(e1, e2);
    }
}
