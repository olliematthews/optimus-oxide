use super::*;
use std::collections::HashSet;
mod tests {
    use crate::tokenizer::encode::{encode, encode_from_merge_tree, vocab_to_merge_tree};

    use super::*;

    #[test]
    fn test_encode() {
        // A simple test to make sure that common words get encoded
        let target_word = "sausages";

        let mut input_str = String::new();
        for char in "abcdefghijklm".chars() {
            input_str.push_str(target_word);
            input_str.push(char);
        }
        let (_, vocab) = bpe_on_str(&input_str, (target_word.len() - 2) as u32).unwrap();

        let vocab_words: HashSet<String> = vocab
            .values()
            .flat_map(|vocab_bytes| str::from_utf8(&vocab_bytes).ok())
            .map(|val| val.to_owned())
            .collect();

        assert!(vocab_words.contains(target_word));
    }

    #[test]
    fn test_bpe_results() {
        let input_file = "./data/botchan.txt";
        // A simple test to make sure that common words get encoded
        let (merges, _) = bpe_on_file(input_file, 10).unwrap();

        let mut input_file_as_str = String::new();
        File::open(input_file)
            .unwrap()
            .read_to_string(&mut input_file_as_str)
            .unwrap();
        let encoded = encode(&input_file_as_str, merges.clone()).unwrap();

        let mut merge_token_occs: HashMap<u16, u16> =
            merges.iter().map(|(_, token)| (token.clone(), 0)).collect();

        for text_char in encoded.iter() {
            if let Some(count) = merge_token_occs.get_mut(text_char) {
                *count += 1;
            }
        }

        let mut maybe_last_word_count = None;
        for (_, merge_token) in merges.iter() {
            let mut token_count = merge_token_occs[merge_token];
            // Count tokens which were merged into something else
            for (other_merge_from, other_merge_token) in merges.iter() {
                if other_merge_from.0 == *merge_token || other_merge_from.1 == *merge_token {
                    token_count += merge_token_occs[other_merge_token];
                }
            }
            if let Some(last_word_count) = maybe_last_word_count {
                assert!(token_count <= last_word_count);
            }
            maybe_last_word_count = Some(token_count);
        }
    }

    #[test]
    fn test_roudtrip() {
        let input_str = "Hi hi, hello silly eggs and sausages and pickle and kettle chips yahooo what a large elephant";
        let (merges, vocab) = bpe_on_str(input_str, 10).unwrap();
        assert_eq!(
            input_str,
            decode(encode(input_str, merges).unwrap(), &vocab).unwrap()
        );
    }

    #[test]
    fn test_fast_encode() {
        let input_file = "./data/botchan.txt";
        // A simple test to make sure that common words get encoded
        let (merges, vocab) = bpe_on_file(input_file, 100).unwrap();

        let mut input_file_as_str = String::new();
        File::open(input_file)
            .unwrap()
            .read_to_string(&mut input_file_as_str)
            .unwrap();
        let encoded = encode(&input_file_as_str, merges.clone()).unwrap();

        let merge_tree = vocab_to_merge_tree(&vocab).unwrap();
        let fast_encoded = encode_from_merge_tree(&input_file_as_str, merge_tree).unwrap();
        let decoded = decode(encoded.clone(), &vocab).unwrap();
        assert_eq!(decoded, input_file_as_str);

        let n_test = 40;
        println!("Encoded: {:?}", &encoded[..n_test]);
        println!("FastEncoded: {:?}", &fast_encoded[..n_test]);
        assert!(encoded[..n_test] == fast_encoded[..n_test]);
    }
}
