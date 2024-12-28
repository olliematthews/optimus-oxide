use crate::do_at_key_with_default;
use core::str;
// This is going to by my tokenizer
use crate::exceptions::TokenizerError;
use crate::tokenizer::utils::to_word_tokens;
use anyhow::Result;
use log::{debug, info};
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::Read;
use tqdm;

pub fn bpe_on_str(
    input_str: &str,
    n_merges: u32,
) -> Result<(Vec<((u16, u16), u16)>, HashMap<u16, Vec<u8>>)> {
    let encoded: Vec<u8> = input_str.bytes().collect();
    bpe(encoded, n_merges)
}

pub fn bpe_on_file(
    input_file: &str,
    n_merges: u32,
) -> Result<(Vec<((u16, u16), u16)>, HashMap<u16, Vec<u8>>)> {
    info!("Starting BPE algorithm on {input_file}");
    let mut f = File::open(input_file)?;

    let mut buffer = Vec::new();
    // read the whole file
    f.read_to_end(&mut buffer)?;
    bpe(buffer, n_merges)
}

pub fn bpe(
    input_bytes: Vec<u8>,
    n_merges: u32,
) -> Result<(Vec<((u16, u16), u16)>, HashMap<u16, Vec<u8>>)> {
    let start_len = input_bytes.len();

    // Split the input string by the split byte. This makes the algorithm run much faster, but means that you cannot have multi-word tokens.
    let split_byte: u8 = ' ' as u8;
    let mut words: Vec<(Vec<u16>, u32)> = to_word_tokens(input_bytes, split_byte);

    // Added tokens start at 256 (max byte value + 1)
    let mut next_token_id: u16 = u8::MAX as u16 + 1;

    let mut merges: Vec<((u16, u16), u16)> = Vec::new();
    let mut vocab: HashMap<u16, Vec<u8>> = HashMap::new();

    // Fill out the initial pairs which occur in the words
    let mut pairs: HashMap<(u16, u16), u32> = HashMap::new(); // pair, and number of occurrences in the input
    let mut words_with_pair: HashMap<(u16, u16), HashSet<usize>> = HashMap::new(); // mapping from pair to words in which it occurs
    for (word_index, (tokens, n_occurrences)) in words.iter().enumerate() {
        for i in 0..tokens.len() - 1 {
            let pair = (tokens[i], tokens[i + 1]);
            do_at_key_with_default!(pairs, &pair, add * n_occurrences, *n_occurrences);
            do_at_key_with_default!(words_with_pair, &pair, insert word_index);
        }
    }
    debug! {"Initial pairs: {:?}", pairs};
    debug! {"Initial wwps: {:?}", words_with_pair};

    let mut merge_iter = 0;
    let mut pbar = tqdm::pbar(Some(n_merges as usize));

    while merge_iter < n_merges {
        let mut merged_tokens: HashSet<u16> = HashSet::new(); // Tokens which have been merged in this pass

        // Get the best pairs
        let mut best_pairs: Vec<(u16, u16)> = Vec::new();
        let mut best_pair_n_matches: u32 = 1;
        for (pair, n_occurrences) in &pairs {
            if *n_occurrences >= best_pair_n_matches {
                if *n_occurrences > best_pair_n_matches {
                    best_pairs.clear();
                    best_pair_n_matches = *n_occurrences;
                }
                best_pairs.push(pair.clone());
            }
        }

        for merge_from_pair in best_pairs {
            if merge_iter >= n_merges {
                break;
            }
            // Run through the pairs which have the same occurrence count and do not share any tokens
            if merged_tokens.contains(&merge_from_pair.0)
                || merged_tokens.contains(&merge_from_pair.1)
            {
                continue;
            }
            merged_tokens.insert(merge_from_pair.0);
            merged_tokens.insert(merge_from_pair.1);

            let merge_to = next_token_id;
            next_token_id += 1;

            // Now add the merge in
            merges.push((merge_from_pair, merge_to));

            // And add the vocab entry in
            let mut reverse_map: Vec<u8> = Vec::new();
            for token in [merge_from_pair.0, merge_from_pair.1] {
                if let Some(submap) = vocab.get(&token) {
                    reverse_map.extend(submap);
                } else {
                    reverse_map.push(token as u8)
                }
            }
            vocab.insert(merge_to, reverse_map);

            // Apply the merge...
            debug! {"Replacing {:?} with {:?}", merge_from_pair, merge_to};

            // Remove the old pair from our pairs
            pairs.remove(&merge_from_pair);
            let words_to_merge = words_with_pair.remove(&merge_from_pair).expect(&format!(
                "This should be in the words_by_pair {merge_from_pair:?}"
            ));
            for word_index in words_to_merge {
                let (ref mut tokens, word_occs) = words
                    .get_mut(word_index)
                    .expect("All word indexes should be valid.");
                // debug! {"Tokens are: {:?}", tokens};

                // First, update the tokens for that word
                let mut pairs_in_word: HashMap<(u16, u16), u32> = HashMap::new(); // Pair, and n_occurrences
                let mut token_idx: usize = 0;
                while token_idx < (tokens.len() - 1) {
                    let pair = (tokens[token_idx], tokens[token_idx + 1]);

                    // If we are not merging this pair, add it to the pair in word count and continue
                    if pair != merge_from_pair {
                        do_at_key_with_default!(pairs_in_word, &pair, add 1, 1);
                        token_idx += 1;
                        continue;
                    }

                    // Apply the merge
                    tokens[token_idx] = merge_to;
                    tokens.remove(token_idx + 1);
                    // debug! {"Tokens after remove are: {:?}", tokens};

                    // Update state to account for pairs being replaced before and after the new token
                    let mut new_pairs: Vec<(u16, u16)> = Vec::new();
                    let mut replaced_pairs: Vec<(u16, u16)> = Vec::new();

                    // Check for a pair replacement after the new pair
                    if token_idx < tokens.len() - 1 {
                        new_pairs.push((merge_to, tokens[token_idx + 1]));
                        let replaced_pair = (pair.1, tokens[token_idx + 1]);
                        replaced_pairs.push(replaced_pair);
                        // If a pair was removed, indicate this
                        if !pairs_in_word.contains_key(&replaced_pair) {
                            pairs_in_word.insert(replaced_pair.clone(), 0);
                        }
                    }
                    // Check for a pair replacement before the new pair
                    if token_idx > 0 {
                        new_pairs.push((tokens[token_idx - 1], merge_to));
                        let replaced_pair = (tokens[token_idx - 1], pair.0);
                        replaced_pairs.push(replaced_pair);
                        // Indicate that an instance of the pair has been removed
                        do_at_key_with_default!(pairs_in_word, &replaced_pair, sub 1, 0);
                    }

                    for new_pair in new_pairs {
                        // Update state
                        do_at_key_with_default!(pairs, &new_pair, add * word_occs, *word_occs);
                        do_at_key_with_default!(words_with_pair, &new_pair, insert word_index);
                        do_at_key_with_default!(pairs_in_word, &new_pair, add 1, 1);
                        // debug!(
                        //     "Added in new pair: {:?} count is now: {:?}. WOK is {}",
                        //     new_pair.clone(),
                        //     pairs.get(&new_pair),
                        // );
                    }
                    for replaced_pair in replaced_pairs {
                        if replaced_pair != merge_from_pair {
                            // Reduce the count on the pair
                            *(pairs
                                .get_mut(&replaced_pair)
                                .expect("This pair must be in there")) -= *word_occs;
                        }
                    }
                }
                // Account for any pairs which have been removed from our word
                for (pair, n_occurrences) in pairs_in_word {
                    if n_occurrences == 0 && pair != merge_from_pair {
                        debug! {"Removing word {word_index} from pair: {:?}", pair.clone()};
                        words_with_pair
                            .get_mut(&pair)
                            .expect(&format!(
                                "We expect pair {pair:?} to be in there before removal"
                            ))
                            .remove(&word_index);
                    }
                }
            }
            merge_iter += 1;
            pbar.update(1)?;
        }
    }
    let compression = (start_len
        - words
            .iter()
            .map(|(tokens, n_occurrences)| tokens.len() * (*n_occurrences as usize))
            .sum::<usize>())
        / start_len;
    info!("Compression of tokens by: {compression:.2}%");
    Ok((merges, vocab))
}

pub fn encode(input_str: &str, merges: Vec<((u16, u16), u16)>) -> Result<Vec<u16>> {
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

    Ok(encoded)
}

pub fn decode(encoded: Vec<u16>, vocab: HashMap<u16, Vec<u8>>) -> Result<String> {
    let decoded: Vec<u8> = encoded
        .iter()
        .map(|element| {
            if *element > 255 {
                Ok(vocab
                    .get(element)
                    .ok_or(TokenizerError::UnrecognizedToken(*element))?
                    .clone())
            } else {
                Ok(vec![*element as u8])
            }
        })
        .collect::<Result<Vec<Vec<u8>>>>()?
        .drain(..)
        .flatten()
        .collect();
    Ok(str::from_utf8(&decoded)?.to_owned())
}

#[cfg(test)]
#[path = "./unit_tests/tokenizer_tests.rs"]
mod tokenizer_tests;
