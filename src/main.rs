use core::str;
// This is going to by my tokenizer
use std::collections::HashMap;
use std::env;
use std::fs;
use transformer_oxide::tokenizer::{bpe, decode, encode};

fn main() {
    let input_str = "Hi hi, hello silly eggs and sausages and pickle and kettle chips yahooo what a large elephant";
    let (merges, vocab) = bpe(input_str, 10);
    print!("{}", decode(encode(input_str, merges), vocab));
}
