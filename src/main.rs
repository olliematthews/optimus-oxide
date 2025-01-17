use core::str;
use std::collections::HashSet;
// This is going to by my tokenizer
use env_logger;

use log::info;
use transformer_oxide::tokenizer::tokenizer::{bpe_on_file, bpe_on_str};

fn main() {
    env_logger::init();

    let (_, vocab) = bpe_on_file("./data/botchan.txt", 1000).unwrap();

    let mut vocab_words: Vec<String> = vocab
        .values()
        .flat_map(|vocab_bytes| str::from_utf8(&vocab_bytes).ok())
        .map(|val| val.to_owned())
        .collect();
    vocab_words.sort_by(|slf, other| slf.len().partial_cmp(&other.len()).unwrap());

    info!("First ten merges: {:?}", &vocab_words[..10]);
    info!(
        "Last ten merges: {:?}",
        &vocab_words[vocab_words.len() - 10..vocab_words.len()]
    );
}
