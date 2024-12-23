use std::collections::HashMap;

pub fn to_word_tokens(input_bytes: Vec<u8>, split_byte: u8) -> HashMap<Vec<u16>, u32> {
    let mut words: HashMap<Vec<u16>, u32> = HashMap::new();
    let mut current_word: Vec<u16> = Vec::new();

    for input_byte in input_bytes {
        if input_byte == split_byte && current_word.len() > 0 {
            if words.contains_key(&current_word) {
                *words.get_mut(&current_word).unwrap() += 1;
            } else {
                words.insert(current_word, 1);
            }
            current_word = Vec::new();
        }
        current_word.push(input_byte as u16);
    }
    if current_word.len() > 0 {
        if words.contains_key(&current_word) {
            *words.get_mut(&current_word).unwrap() += 1;
        } else {
            words.insert(current_word, 1);
        }
    }
    words
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_split() {
        // Make sure our splitting function works ok
        let split_byte: u8 = ' ' as u8;

        // Check a repeated string with a space before
        let n_repeats: u32 = 6;
        let input_to_repeat = " hi";
        let input_str = input_to_repeat.repeat(n_repeats as usize);
        let word_tokens = to_word_tokens(input_str.bytes().collect(), split_byte);
        assert_eq!(word_tokens.len(), 1);
        let word_bytes: Vec<u16> = input_to_repeat.bytes().map(|v| v as u16).collect();
        assert_eq!(word_tokens[&(word_bytes)], n_repeats);

        // Check another word gets added in ok
        let input_str = format!("hi{input_str}");
        assert_eq!(
            to_word_tokens(input_str.bytes().collect(), split_byte).len(),
            2
        );
    }
}
