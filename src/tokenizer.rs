use core::str;
// This is going to by my tokenizer
use log::{debug, info};
use std::collections::HashMap;
use std::fs::File;
use std::io::Read;

pub fn bpe_on_str(
    input_str: &str,
    n_merges: u32,
) -> (Vec<((u16, u16), u16)>, HashMap<u16, Vec<u8>>) {
    let encoded: Vec<u8> = input_str.bytes().collect();
    bpe(encoded, n_merges)
}

pub fn bpe_on_file(
    input_file: &str,
    n_merges: u32,
) -> (Vec<((u16, u16), u16)>, HashMap<u16, Vec<u8>>) {
    info!("Starting BPE algorithm on {input_file}");
    let mut f = File::open(input_file).unwrap();

    let mut buffer = Vec::new();
    // read the whole file
    f.read_to_end(&mut buffer).unwrap();
    bpe(buffer, n_merges)
}

pub fn bpe(input_bytes: Vec<u8>, n_merges: u32) -> (Vec<((u16, u16), u16)>, HashMap<u16, Vec<u8>>) {
    let mut next_count: u16 = u8::MAX as u16 + 1;
    let mut encoded: Vec<u16> = input_bytes.iter().map(|v| *v as u16).collect();

    let mut merges: Vec<((u16, u16), u16)> = Vec::new();
    let mut vocab: HashMap<u16, Vec<u8>> = HashMap::new();

    let start_len = encoded.len();

    for merge in 0..n_merges {
        // Find the most commonly occuring pair
        let mut pairs: HashMap<(u16, u16), u32> = HashMap::new();
        let mut best_pair: Option<((u16, u16), u32)> = None;
        for i in 0..encoded.len() - 1 {
            let key = (encoded[i], encoded[i + 1]);
            if let Some(value) = pairs.get_mut(&key) {
                *value += 1;
                if let Some(bp) = best_pair {
                    if *value > bp.1 {
                        best_pair = Some((key, *value));
                    }
                } else {
                    best_pair = Some((key, *value));
                }
            } else {
                pairs.insert(key, 1);
            }
        }

        let merge_from = best_pair.unwrap().0;
        let merge_to = next_count;
        next_count += 1;

        // Now add the merge in
        merges.push((merge_from, merge_to));

        // And add the vocab entry in
        let mut reverse_map: Vec<u8> = Vec::new();
        if let Some(submap) = vocab.get(&merge_from.0) {
            reverse_map.extend(submap);
        } else {
            reverse_map.push(merge_from.0 as u8)
        }
        if let Some(submap) = vocab.get(&merge_from.1) {
            reverse_map.extend(submap);
        } else {
            reverse_map.push(merge_from.1 as u8)
        }

        vocab.insert(merge_to, reverse_map);

        // And apply the merge...
        let mut i: usize = 0;
        while i < (encoded.len() - 1) {
            if (encoded[i], encoded[i + 1]) == merge_from {
                encoded[i] = merge_to;
                encoded.remove(i + 1);
            } else {
                i += 1;
            }
        }
        debug!("MERGE: {}. Len: {}", merge, encoded.len());
    }
    let compression = (start_len - encoded.len()) / start_len;
    info!("Compression of tokens by: {compression:.2}%");
    (merges, vocab)
}

pub fn encode(input_str: &str, merges: Vec<((u16, u16), u16)>) -> Vec<u16> {
    let mut encoded: Vec<u16> = input_str
        .bytes()
        .map(|val: u8| -> u16 { val as u16 })
        .collect();

    for (merge_from, merge_to) in merges {
        let mut i: usize = 0;
        while i < (encoded.len() - 1) {
            if (encoded[i], encoded[i + 1]) == merge_from {
                encoded[i] = merge_to;
                encoded.remove(i + 1);
            } else {
                i += 1;
            }
        }
    }

    encoded
}

pub fn decode(encoded: Vec<u16>, vocab: HashMap<u16, Vec<u8>>) -> String {
    let decoded: Vec<u8> = encoded
        .iter()
        .flat_map(|element| {
            if *element > 255 {
                vocab[element].clone()
            } else {
                vec![*element as u8]
            }
        })
        .collect();
    str::from_utf8(&decoded).unwrap().to_owned()
}

#[cfg(test)]
#[path = "./unit_tests/tokenizer_tests.rs"]
mod tokenizer_tests;
