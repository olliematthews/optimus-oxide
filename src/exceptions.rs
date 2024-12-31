use thiserror::Error;

#[derive(Error, Debug)]
pub enum TokenizerError {
    #[error("token {0} not found in the vocab.")]
    UnrecognizedToken(u16),
    #[error("unknown tokenizer error")]
    Unknown,
}

#[derive(Error, Debug)]
pub enum MatrixError {
    #[error("{0}")]
    MatrixError(String),
    #[error("unknown tokenizer error")]
    Unknown,
}
