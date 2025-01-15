use anyhow::{bail, Result};
use std::collections::HashMap;

#[derive(Default)]
pub struct MergeTree {
    children: HashMap<u8, MergeTree>,
    token: Option<u16>,
}
// enum MergeTreeElement {
//     Node(MergeTreeNode),
//     Token(u16),
// }

impl MergeTree {
    // fn new_node() -> MergeTreeElement {
    //     MergeTreeElement::Node(Default::default())
    // }

    // fn new_token(val: u16) -> MergeTreeElement {
    //     MergeTreeElement::Token(val)
    // }
    fn get(&self, key: &u8) -> Option<&MergeTree> {
        self.children.get(key)
    }

    fn get_mut(&mut self, key: &u8) -> Option<&mut MergeTree> {
        self.children.get_mut(&key)
    }

    fn _insert_and_get_mut_ref(&mut self, key: &u8, val: MergeTree) -> &mut MergeTree {
        self.children.insert(*key, val);
        self.get_mut(key).expect("We just added this element in")
    }

    fn _insert(&mut self, key: &u8, val: MergeTree) {
        self.children.insert(*key, val);
    }

    fn has_child(&self, key: &u8) -> bool {
        self.children.contains_key(key)
    }

    fn set(&mut self, keys: &Vec<u8>, token: u16) -> Result<()> {
        let mut current_node = self;

        if keys.len() == 0 {
            bail!("Path is empty. Cannot set a value at the root.");
        }
        for key in keys {
            if !current_node.has_child(key) {
                current_node._insert(key, Default::default());
            }
            current_node = current_node.get_mut(key).expect("We just checked this");
        }
        if let Some(old_token) = current_node.token {
            bail!(
                "Path is not empty. Found token {} at path {:?}",
                old_token,
                keys
            );
        }
        current_node.token = Some(token);
        return Ok(());
    }
}

pub fn vocab_to_merge_tree(vocab: &HashMap<u16, Vec<u8>>) -> Result<MergeTree> {
    let mut root: MergeTree = Default::default();
    for (token, chars) in vocab.iter() {
        root.set(chars, *token)?;
    }
    Ok(root)
}

pub fn encode_from_merge_tree(input_str: &str, merge_tree: MergeTree) -> Result<Vec<u16>> {
    let encoded: &[u8] = input_str.as_bytes();

    let mut i: usize = 0;
    let mut tokens: Vec<u16> = vec![];
    while i < (encoded.len() - 1) {
        let mut last_valid_token: u16 = encoded[i].into();
        let mut last_valid_j = i;
        if let Some(element) = &merge_tree.get(&encoded[i]) {
            let mut current_node = *element;
            let mut j = i + 1;
            while j < encoded.len() {
                if let Some(child_element) = current_node.get(&encoded[j]) {
                    current_node = child_element;
                } else {
                    break;
                }
                if let Some(token) = current_node.token {
                    last_valid_token = token;
                    last_valid_j = j;
                }
                j += 1;
            }
        }
        tokens.push(last_valid_token);
        i = last_valid_j + 1;
    }
    Ok(tokens)
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
