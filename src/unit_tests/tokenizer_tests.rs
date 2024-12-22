use super::*;
use std::collections::HashSet;
mod tests {
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
        let (_, vocab) = bpe(&input_str, (target_word.len() - 2) as u32);

        let vocab_words: HashSet<String> = vocab
            .values()
            .flat_map(|vocab_bytes| str::from_utf8(&vocab_bytes).ok())
            .map(|val| val.to_owned())
            .collect();

        assert!(vocab_words.contains(target_word));
    }

    #[test]
    fn test_roudtrip() {
        let input_str = "Hi hi, hello silly eggs and sausages and pickle and kettle chips yahooo what a large elephant";
        let (merges, vocab) = bpe(input_str, 10);
        assert_eq!(input_str, decode(encode(input_str, merges), vocab));
    }
}
