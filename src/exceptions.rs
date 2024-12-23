use thiserror::Error;

#[derive(Error, Debug)]
pub enum TokenizerError {
    #[error("token {0} not found in the vocab.")]
    UnrecognizedToken(u16),
    #[error("unknown tokenizer error")]
    Unknown,
}
