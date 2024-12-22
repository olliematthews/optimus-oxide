use core::str;
// This is going to by my tokenizer
use env_logger;
use log::info;
use transformer_oxide::tokenizer::bpe_on_file;

fn main() {
    env_logger::init();
    let (_, vocab) = bpe_on_file("./data/botchan.txt", 200);

    let mut vocab_words: Vec<String> = vocab
        .values()
        .flat_map(|vocab_bytes| str::from_utf8(&vocab_bytes).ok())
        .map(|val| val.to_owned())
        .collect();
    vocab_words.sort_by(|slf, other| slf.len().partial_cmp(&other.len()).unwrap());

    info!("{:?}", &vocab_words[..10]);
}
