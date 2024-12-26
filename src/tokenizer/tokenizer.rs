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

    let mut next_count: u16 = u8::MAX as u16 + 1;

    let mut merges: Vec<((u16, u16), u16)> = Vec::new();
    let mut vocab: HashMap<u16, Vec<u8>> = HashMap::new();

    let mut merge_iter = 0;
    let mut pbar = tqdm::pbar(Some(n_merges as usize));

    // Find the most commonly occurring pair
    let mut pairs: HashMap<(u16, u16), u32> = HashMap::new();
    let mut words_by_pair_occurrance: HashMap<(u16, u16), HashSet<usize>> = HashMap::new();

    for (word_index, (tokens, n_occurrances)) in words.iter().enumerate() {
        for i in 0..tokens.len() - 1 {
            let pair = (tokens[i], tokens[i + 1]);
            if let Some(value) = pairs.get_mut(&pair) {
                *value += *n_occurrances;
            } else {
                pairs.insert(pair, *n_occurrances);
            }

            if let Some(word_list) = words_by_pair_occurrance.get_mut(&pair) {
                word_list.insert(word_index);
            } else {
                debug!("INSERTED into WBP: {:?}", pair.clone());
                words_by_pair_occurrance.insert(pair, HashSet::from([word_index]));
            }
        }
    }

    while merge_iter < n_merges {
        let mut merged_tokens: HashSet<u16> = HashSet::new(); // Keep track of what you have merged already

        // Get the best pairs
        // TODO: do this with a more optimal pair structure
        let mut best_pairs: Vec<(u16, u16)> = Vec::new();
        let mut best_pair_n_matches: u32 = 1;
        for (pair, n_occurrances) in &pairs {
            if *n_occurrances >= best_pair_n_matches {
                if *n_occurrances > best_pair_n_matches {
                    best_pairs.clear();
                    best_pair_n_matches = *n_occurrances;
                }
                best_pairs.push(pair.clone());
            }
        }

        for merge_from_pair in best_pairs {
            // Make sure not to try and merge a pair which includes any tokens which have already been merged
            if merge_iter >= n_merges {
                break;
            }
            if merged_tokens.contains(&merge_from_pair.0)
                || merged_tokens.contains(&merge_from_pair.1)
            {
                continue;
            }
            merged_tokens.insert(merge_from_pair.0);
            merged_tokens.insert(merge_from_pair.1);
            let merge_to = next_count;
            next_count += 1;

            // Now add the merge in
            merges.push((merge_from_pair, merge_to));

            // And add the vocab entry in
            let mut reverse_map: Vec<u8> = Vec::new();
            if let Some(submap) = vocab.get(&merge_from_pair.0) {
                reverse_map.extend(submap);
            } else {
                reverse_map.push(merge_from_pair.0 as u8)
            }
            if let Some(submap) = vocab.get(&merge_from_pair.1) {
                reverse_map.extend(submap);
            } else {
                reverse_map.push(merge_from_pair.1 as u8)
            }

            vocab.insert(merge_to, reverse_map);

            // And apply the merge...
            debug!("REMOVING from WBP: {:?}", merge_from_pair.clone());
            let words_to_merge =
                words_by_pair_occurrance
                    .remove(&merge_from_pair)
                    .expect(&format!(
                        "This should be in the words_by_pair {merge_from_pair:?}"
                    ));
            for word_index in words_to_merge {
                debug! {"Replacing {:?} with {:?}", merge_from_pair, merge_to};
                let (ref mut tokens, word_occurrances) = words
                    .get_mut(word_index)
                    .expect("All word indexes should be valid.");
                debug! {"Tokens are: {:?}", tokens};

                // First, update the tokens for that word
                let mut word_pair_occurances: HashMap<(u16, u16), u32> = HashMap::new();
                let mut i: usize = 0;
                while i < (tokens.len() - 1) {
                    let pair = (tokens[i], tokens[i + 1]);
                    if pair == merge_from_pair {
                        tokens[i] = merge_to;
                        tokens.remove(i + 1);
                        debug! {"Tokens after remove are: {:?}", tokens};

                        // Add in the new next pair if there is one
                        if i < tokens.len() - 1 {
                            let new_pair = (merge_to, tokens[i + 1]);

                            // Add the new pair to the set of pairs
                            if let Some(pair_occurrances) = pairs.get_mut(&new_pair) {
                                *pair_occurrances += *word_occurrances;
                            } else {
                                pairs.insert(new_pair, *word_occurrances);
                            }
                            // ...and into the word occurances
                            if let Some(pair_occurrances) = word_pair_occurances.get_mut(&new_pair)
                            {
                                *pair_occurrances += 1;
                            } else {
                                word_pair_occurances.insert(new_pair, 1);
                            }
                            // ... and into the words_by_pair
                            if let Some(words) = words_by_pair_occurrance.get_mut(&new_pair) {
                                words.insert(word_index);
                            } else {
                                words_by_pair_occurrance
                                    .insert(new_pair.clone(), HashSet::from([word_index]));
                            }

                            // Now consider the pair we just replaced behind
                            let replaced_pair = (pair.1, tokens[i + 1]);
                            debug!(
                                "Removing pair {:?} which occurs {:?} times",
                                replaced_pair,
                                pairs.get(&replaced_pair).unwrap()
                            );

                            // Reduce the count on the pair
                            *(pairs
                                .get_mut(&replaced_pair)
                                .expect("This pair must be in there")) -= 1;
                            // Inidicate that an instance of the pair has been removed
                            if !word_pair_occurances.contains_key(&replaced_pair) {
                                // Use this to indicate later that this pair should be removed
                                word_pair_occurances.insert(replaced_pair.clone(), 0);
                            }
                            debug!(
                                "Added in new pair: {:?} count is now: {:?}. WOK is {}",
                                new_pair.clone(),
                                pairs.get(&new_pair),
                                word_occurrances.clone()
                            );
                        }
                        // Add in the new previous pair if there is one
                        if i > 0 {
                            let new_pair = (tokens[i - 1], merge_to);

                            // Add the new pair into the global pairs
                            if let Some(pair_occurrances) = pairs.get_mut(&new_pair) {
                                *pair_occurrances += *word_occurrances;
                            } else {
                                pairs.insert(new_pair, *word_occurrances);
                            }
                            // ...and into the word occurances
                            if let Some(pair_occurrances) = word_pair_occurances.get_mut(&new_pair)
                            {
                                *pair_occurrances += 1;
                            } else {
                                word_pair_occurances.insert(new_pair, 1);
                            }
                            // ... and into the words_by_pair
                            if let Some(words) = words_by_pair_occurrance.get_mut(&new_pair) {
                                words.insert(word_index);
                            } else {
                                words_by_pair_occurrance
                                    .insert(new_pair.clone(), HashSet::from([word_index]));
                            }

                            // Consider the replaced pair
                            let replaced_pair = (tokens[i - 1], pair.0);
                            // debug!(
                            //     "Removing pair {:?} which occurs {:?} times",
                            //     replaced_pair, pairs[&replaced_pair]
                            // );
                            // Decrement the count
                            *(pairs.get_mut(&replaced_pair).expect(&format!(
                                "This pair must be in there: {replaced_pair:?}"
                            ))) -= 1;

                            // Reduce the occurrance count in this word
                            if let Some(pair_occurances) =
                                word_pair_occurances.get_mut(&replaced_pair)
                            {
                                *pair_occurances -= 1;
                            } else {
                                word_pair_occurances.insert(replaced_pair, 0);
                            }

                            debug!(
                                "Added in new pair: {:?} count is now: {:?}. WOK is {}",
                                new_pair.clone(),
                                pairs.get(&new_pair),
                                word_occurrances.clone()
                            );
                        }
                    } else {
                        if let Some(pair_occurrances) = word_pair_occurances.get_mut(&pair) {
                            *pair_occurrances += 1;
                        } else {
                            word_pair_occurances.insert(pair, 1);
                        }
                        i += 1;
                    }
                }
                // Make sure to account for any removed pairs in our words_by_pair_occurrances mapping
                for (pair, n_occurrances) in word_pair_occurances {
                    if n_occurrances == 0 {
                        debug! {"Removing word {word_index} from pair: {:?}", pair.clone()};
                        words_by_pair_occurrance
                            .get_mut(&pair)
                            .expect("We expect this to be in there before removal")
                            .remove(&word_index);
                    }
                }
            }
            pairs.remove(&merge_from_pair);
            merge_iter += 1;
            pbar.update(1)?;
        }
    }
    let compression = (start_len
        - words
            .iter()
            .map(|(tokens, n_occurrances)| tokens.len() * (*n_occurrances as usize))
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
#[path = "./tokenizer_tests.rs"]
mod tokenizer_tests;
