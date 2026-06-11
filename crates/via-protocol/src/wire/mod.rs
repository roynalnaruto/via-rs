//! Wire types: queries, answers, keys, and the `WireFormat` trait.
//!
//! Everything here is re-exported at the crate root; consumers import
//! `via_protocol::{…}`, not `via_protocol::wire::…`.

pub mod answer;
pub mod format;
pub mod keys;
pub mod query;

pub use answer::CompressedAnswer;
pub use format::{PrgCompressed, Uncompressed, WireFormat};
pub use keys::{PublicParams, QueryCompressionKey};
#[cfg(feature = "via-b")]
pub use query::BatchedQuery;
pub use query::{CompressedQuery, DecompressedQuery};
